'''
COLOR-CERT main code

The code heritates heavily from: 
[1] Interactive Deep Colorization](https://github.com/junyanz/interactive-deep-colorization) 
[2] Macer(https://github.com/RuntianZ/macer).
[3] https://github.com/locuslab/smoothing/blob/master/code/certify.py
'''

import torch
import torch.nn.functional as F
from torch.distributions.normal import Normal
from color_model import color_net
from rs.certify import certify_gen
import argparse
import time
import os
import utils
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10
from architectures import resnet110, get_sigma_gen
import analyze
from tqdm import tqdm

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0)

def scaler(x, cmodel, k):
    '''
    Convert image to LAB, and choose hints given clusters
    '''
    data = {}
    data_lab = utils.rgb2lab(x)
    data['L'] = data_lab[:,[0,],:,:]
    data['AB'] = data_lab[:,1:,:,:]
    hints = data['AB'].clone()
    mask = torch.full_like(data['L'], -.5)

    with torch.no_grad():      
        N,C,H,W = hints.shape
        # Calculate entropy based on colorization model. Use 0 hints for this
        logits, reg = cmodel(data['L'], torch.zeros_like(hints).cuda(), torch.full_like(mask, -.5).cuda())
        hints_distr = F.softmax(logits, dim=1).clamp(1e-8)
        #print('hints_distr', hints_distr.shape)
        entropy = (-1 * hints_distr * torch.log(hints_distr)).sum(1, keepdim=True)
        #print('entropy', entropy.shape)
        # upsample to original resolution
        entropy = F.interpolate(entropy, scale_factor=(4,4), mode='nearest').view(-1)
    return entropy.reshape(N,1,H,W)

def compute_entropy(x, cmodel, k):
    with torch.no_grad():
        return scaler(x, cmodel, k)

def train_generator_step(epoch, step, orig_inputs, targets, lbd1, lbd2, gauss_num, beta, gamma, num_classes, generator, model, cmodel, m, optimizer, device, verbose=1):
    model.eval()
    generator.train()

    input_size = len(orig_inputs)
    img_size = len(orig_inputs[0].view(-1))
    new_shape = [input_size * gauss_num]
    new_shape.extend(orig_inputs[0].shape)
    orig_sigma = generator(orig_inputs)
    resized_imgs = transforms.Resize((128, 128))(orig_inputs)
    entropy = compute_entropy(resized_imgs, cmodel, k=8)
    entropy = transforms.Resize((orig_inputs[0].shape[-1], orig_inputs[0].shape[-1]))(entropy)
    if verbose and (epoch-1) % 10 == 0 and step == 0:
        analyze.sigma_heatmap(epoch, step, orig_inputs.permute(0,2,3,1).detach().cpu().numpy(), 
            orig_sigma.permute(0,2,3,1).detach().cpu().numpy(), entropy.permute(0,2,3,1).detach().cpu().numpy())
    inputs = orig_inputs.repeat((1, gauss_num, 1, 1)).view(new_shape)
    outputs = model(inputs + torch.randn_like(inputs, device=device) * orig_sigma.repeat((1, gauss_num, 1, 1)).view(new_shape))
    outputs = outputs.reshape((input_size, gauss_num, num_classes))

    # Classification loss
    outputs_softmax = F.softmax(outputs, dim=2).mean(1)
    outputs_logsoftmax = torch.log(outputs_softmax + 1e-10)  # avoid nan
    classification_loss = F.nll_loss(
        outputs_logsoftmax, targets, reduction='sum')

    # Robustness loss
    beta_outputs = outputs * beta  # only apply beta to the robustness loss
    beta_outputs_softmax = F.softmax(beta_outputs, dim=2).mean(1)
    top2 = torch.topk(beta_outputs_softmax, 2)
    top2_score = top2[0]
    top2_idx = top2[1]
    indices_correct = (top2_idx[:, 0] == targets)
    
    out0, out1 = top2_score[indices_correct,
                            0], top2_score[indices_correct, 1]
    robustness_loss = m.icdf(out1) - m.icdf(out0)
    indices = ~torch.isnan(robustness_loss) & ~torch.isinf(robustness_loss) & (torch.abs(robustness_loss) <= gamma)  # hinge
    out0, out1 = out0[indices], out1[indices]

    prod = torch.prod((orig_sigma.view(input_size, -1)) ** (1/img_size), dim=1)[indices_correct]
    robustness_loss = gamma - prod[indices] * (m.icdf(out0) - m.icdf(out1))
    robustness_loss = robustness_loss.sum()
    
    reg_entropy = (entropy.max()-entropy) / (entropy.max()-entropy.min() + 1e-9) # reverse small and large entropy, output value in [0, 1]
    reg_loss = (reg_entropy.detach() * orig_sigma).norm(p=2, dim=(-3,-2,-1))[indices_correct][indices]
    reg_indices = ~torch.isnan(reg_loss) & ~torch.isinf(reg_loss)
    reg_loss = reg_loss[reg_indices].sum()

    # Final objective function
    loss = classification_loss + lbd1 * robustness_loss + lbd2 * reg_loss
    loss /= input_size
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss.detach(), classification_loss.detach(), robustness_loss.detach(), reg_loss.detach(), orig_sigma.detach()


def train_classifier_step(epoch, step, orig_inputs, targets, lbd1, gauss_num, beta, gamma, num_classes, generator, model, cmodel, m, optimizer, device):
    generator.eval()
    model.train()

    input_size = len(orig_inputs)
    img_size = len(orig_inputs[0].view(-1))
    new_shape = [input_size * gauss_num]
    new_shape.extend(orig_inputs[0].shape)
    inputs = orig_inputs.repeat((1, gauss_num, 1, 1)).view(new_shape)
    orig_sigma = generator(orig_inputs)
    outputs = model(inputs + torch.randn_like(inputs, device=device) * orig_sigma.repeat((1, gauss_num, 1, 1)).view(new_shape))
    outputs = outputs.reshape((input_size, gauss_num, num_classes))

    # Classification loss
    outputs_softmax = F.softmax(outputs, dim=2).mean(1)
    outputs_logsoftmax = torch.log(outputs_softmax + 1e-10)  # avoid nan
    classification_loss = F.nll_loss(
        outputs_logsoftmax, targets, reduction='sum')

    # Robustness loss
    beta_outputs = outputs * beta  # only apply beta to the robustness loss
    beta_outputs_softmax = F.softmax(beta_outputs, dim=2).mean(1)
    top2 = torch.topk(beta_outputs_softmax, 2)
    top2_score = top2[0]
    top2_idx = top2[1]
    indices_correct = (top2_idx[:, 0] == targets)  # G_theta

    out0, out1 = top2_score[indices_correct,
                            0], top2_score[indices_correct, 1]
    robustness_loss = m.icdf(out1) - m.icdf(out0)
    indices = ~torch.isnan(robustness_loss) & ~torch.isinf(
        robustness_loss) & (torch.abs(robustness_loss) <= gamma)  # hinge
    out0, out1 = out0[indices], out1[indices]
    # sigma shape 64 x 16
    prod = torch.prod((orig_sigma.view(input_size, -1)) ** (1/img_size), dim=1)[indices_correct]
    robustness_loss = gamma - prod[indices] * (m.icdf(out0) - m.icdf(out1))
    robustness_loss = robustness_loss.sum()

    # Final objective function
    loss = classification_loss + lbd1 * robustness_loss
    loss /= input_size
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss.detach(), classification_loss.detach(), robustness_loss.detach()


def main(args):
    device = torch.device("cuda:0")
    ngpu = torch.cuda.device_count()

    gaussian = Normal(torch.tensor([0.0]).to(device),
                torch.tensor([1.0]).to(device))

    ckptdir = None if args.ckptdir == 'none' else args.ckptdir
    matdir = None if args.matdir == 'none' else args.matdir
    if matdir is not None and not os.path.isdir(matdir):
        os.makedirs(matdir)
    if ckptdir is not None and not os.path.isdir(ckptdir):
        os.makedirs(ckptdir)
    f_checkpoint = None if args.resume_f_ckpt == 'none' else args.resume_f_ckpt
    g_checkpoint = None if args.resume_g_ckpt == 'none' else args.resume_g_ckpt

    # Load dataset and build models
    colorModel = color_net().cuda()
    colorModel.load_state_dict(torch.load('./colorization_model/pytorch.pth'))

    if args.dataset == 'cifar10':
        base_generator = get_sigma_gen(args.min_sigma, args.max_sigma, dataset='cifar10')
        base_model = resnet110()
        
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
        ])

        trainset = CIFAR10(
            root=args.root, train=True, download=True, transform=transform_train)
        testset = CIFAR10(
            root=args.root, train=False, download=True, transform=transform_test)
        num_classes = 10
        trainloader = torch.utils.data.DataLoader(
            trainset, batch_size=args.batch_size, shuffle=True, num_workers=1, pin_memory=True)

        base_model.apply(weights_init)
        base_generator.apply(weights_init)
        # Resume from checkpoint if required
        if f_checkpoint is not None:
            print('==> Resuming classifier from checkpoint..')
            print(f_checkpoint)
            f_checkpoint = torch.load(f_checkpoint)
            base_model.load_state_dict(f_checkpoint['net'])

        start_epoch = 0
        if g_checkpoint is not None:
            print('==> Resuming generator from checkpoint..')
            print(g_checkpoint)
            g_checkpoint = torch.load(g_checkpoint)
            base_generator.load_state_dict(g_checkpoint['net'])
            start_epoch = g_checkpoint['epoch']

        # Data parallel
        model = torch.nn.DataParallel(base_model.to(device), list(range(ngpu)))
        colorModel = torch.nn.DataParallel(colorModel.to(device), list(range(ngpu)))
        colorModel.eval()
        generator = torch.nn.DataParallel(base_generator.to(device), list(range(ngpu)))

        optimizer_f = optim.SGD(
            model.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
        optimizer_g = optim.SGD(
            generator.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    
        scheduler_f = MultiStepLR(
            optimizer_f, milestones=[100, 200], gamma=0.1)
        scheduler_g = MultiStepLR(
            optimizer_g, milestones=[100, 200], gamma=0.1)

        scheduler_f.step(start_epoch)
        scheduler_g.step(start_epoch)

    else:
        raise NotImplementedError

    if not torch.cuda.is_available():
        raise NotImplementedError

    # Main routine
    if args.task == 'train':
        # Training routine
        for epoch in range(start_epoch + 1, args.epochs + 1):
            print('===train(epoch={})==='.format(epoch))
            t1 = time.time()
            loss_total = {'f': 0.0, 'g': 0.0}
            cl_total = {'f': 0.0, 'g': 0.0}
            rl_total = {'f': 0.0, 'g': 0.0}
            reg_total = {'f': 'N/A', 'g': 0.0}
            input_total = 0
            for i, (inputs, targets) in enumerate(tqdm(trainloader)):
                inputs, targets = inputs.to(device), targets.to(device)
                input_size = len(inputs)
                input_total += input_size
                # Enter training later for g
                if epoch >= 200:
                    g_loss, gc, gr, greg, sigma = train_generator_step(epoch, i, inputs, targets, args.lbd1, args.lbd2, 
                        args.gauss_num, args.beta, args.gamma, num_classes, 
                        generator, model, colorModel, gaussian, optimizer_g, device, verbose=1)
                    
                    loss_total['g'] += g_loss.item()
                    cl_total['g'] += gc.item()
                    rl_total['g'] += gr.item()
                    reg_total['g'] += greg.item()

                f_loss, fc, fr = train_classifier_step(epoch, i, inputs, targets, args.lbd1, 
                    args.gauss_num, args.beta, args.gamma, num_classes,
                    generator, model, colorModel, gaussian, optimizer_f, device)
                loss_total['f'] += f_loss.item()
                cl_total['f'] += fc.item()
                rl_total['f'] += fr.item()
                
            # log down and print losses
            loss_total['f'] /= input_total
            cl_total['f'] /= input_total
            rl_total['f'] /= input_total
            loss_total['g'] /= input_total
            cl_total['g'] /= input_total
            rl_total['g'] /= input_total
            reg_total['g'] /= input_total
            print(f"Total loss: (f) {loss_total['f']}, (g) {loss_total['g']} | Classification Loss: (f) {cl_total['f']}, (g) {cl_total['g']} | Robustness Loss: (f) {rl_total['f']}, (g) {rl_total['g']} | Reg loss: (g) {reg_total['g']}")
            
            scheduler_f.step()
            scheduler_g.step()
            t2 = time.time()
            print('Elapsed time: {}'.format(t2 - t1))

            if epoch % 10 == 0 and epoch >= 200:
                # Certify test
                print('===test(epoch={})==='.format(epoch))
                t1 = time.time()
                model.eval()
                generator.eval()
                certify_gen(model, generator, device, testset, transform_test, num_classes,
                            experiment_name=f'{args.dataset}-{args.task}-c_aug_off',
                            mode='hard', start_img=args.start_img, num_img=args.num_img, 
                            beta=args.beta, 
                            matfile=(None if matdir is None else os.path.join(matdir, '{}.mat'.format(epoch))))
                t2 = time.time()
                print('Elapsed time: {}'.format(t2 - t1))

                if ckptdir is not None:
                    # Save checkpoint
                    print('==> Saving {}.pth..'.format(epoch))
                    try:
                        f_state = {
                            'net': base_model.state_dict(),
                            'epoch': epoch,
                            }
                        torch.save(f_state, '{}/{}_f.pth'.format(ckptdir, epoch))
                        g_state = {
                            'net': base_generator.state_dict(),
                            'epoch': epoch,
                            }
                        torch.save(g_state, '{}/{}_g.pth'.format(ckptdir, epoch))
                    except OSError:
                        print('OSError while saving {}.pth'.format(epoch))
                        print('Ignoring...')

    else:
        # Test routine
        model.eval()
        generator.eval()
        certify_gen(model, generator, device, testset, transform_test, num_classes,
                    experiment_name=f'{args.dataset}-{args.task}-c_aug_off',
                    mode='both', start_img=args.start_img, num_img=args.num_img, skip=args.skip,
                    beta=args.beta,
                    matfile=(None if matdir is None else os.path.join(matdir, '{}.mat'.format(start_epoch))))

    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Color-cert Train and Test')
    parser.add_argument('--task', default='train',
                        type=str, help='Task: train or test')
    parser.add_argument('--root', default='dataset', type=str, help='Dataset path')
    parser.add_argument('--dataset', default='cifar10', type=str, help='Dataset')
    parser.add_argument('--resume_f_ckpt', default='none', type=str,
                        help='Classifier checkpoint path to resume')
    parser.add_argument('--resume_g_ckpt', default='none', type=str,
                        help='Generator checkpoint path to resume')
    parser.add_argument('--ckptdir', default='none', type=str,
                        help='Checkpoints save directory')
    parser.add_argument('--matdir', default='none', type=str,
                        help='Matfiles save directory')

    parser.add_argument('--epochs', default=440,
                        type=int, help='Number of training epochs')
    parser.add_argument('--gauss_num', default=16, type=int,
                        help='Number of Gaussian samples per input')
    parser.add_argument('--batch_size', default=32, type=int, help='Batch size')

    # params for train
    parser.add_argument('--lr', default=0.01, type=float, help='Initial learning rate')
    parser.add_argument('--min_sigma', default=0.5, type=float,
                        help='Minimal std of generated gaussian noise (also used in test)')
    parser.add_argument('--max_sigma', default=0.5, type=float,
                        help='Maximal std of generated gaussian noise (also used in test)')
    parser.add_argument('--lbd1', default=6.0, type=float,
                        help='Weight of robustness loss')
    parser.add_argument('--lbd2', default=1.0, type=float,
                        help='Weight of robustness loss')
    parser.add_argument('--gamma', default=8.0, type=float,
                        help='Hinge factor')
    parser.add_argument('--beta', default=16.0, type=float,
                        help='Inverse temperature of softmax (also used in test)')               

    # params for test
    parser.add_argument('--start_img', default=0,
                        type=int, help='Image index to start (choose it randomly)')
    parser.add_argument('--num_img', default=500, type=int,
                        help='Number of test images')
    parser.add_argument('--skip', default=1, type=int,
                        help='Number of skipped images per test image')

    args = parser.parse_args()
    print("Experiment Name: Train sigma generator and base classifier (resnet110) in an adversarial way (penalize low entropy).")
    print(f"Hyper-prameters:\n\
            prod = torch.prod((orig_sigma.view(input_size, -1)) ** (1/img_size), dim=1)[indices_correct]\n\
            min_sigma = torch.min(orig_sigma.view(input_size,-1), dim=-1).values[indices_correct]\n\
            robustness_loss = gamma - prod[indices] * (m.icdf(out0) - m.icdf(out1))\n\
            robustness_loss = robustness_loss.sum()\n\
            reg_entropy = (entropy.max()-entropy) / (entropy.max()-entropy.min())\n\
            reg_loss = (reg_entropy*orig_sigma).norm(p=2, dim=(-3,-2,-1))[indices_correct][indices].mean()\n\
            g: loss = classification_loss + lbd1 * robustness_loss + lbd2 * reg_loss\n\
            f: loss = classification_loss + lbd1 * robustness_loss\n\
            {args}")

    main(args)
    
