import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import math

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

def get_sigma_gen(min_sigma, max_sigma, dataset) -> torch.nn.Module:
    """ Return a neural network (with random weights)
    :param arch: the architecture
    :param dataset: the dataset - should be in the datasets.DATASETS list
    :return: a Pytorch module
    """
    if dataset == 'cifar10':
        model = Generator(min_sigma, max_sigma).cuda()
    else:
        raise NotImplementedError("Only implemented for CIFAR10")
    model.apply(weights_init)
    cudnn.benchmark = True
    return model

def resnet110(**kwargs):
  model = ResNet_Cifar(BasicBlock, [18, 18, 18], width=1, **kwargs)
  return model


def conv3x3(in_planes, out_planes, stride=1):
  " 3x3 convolution with padding "
  return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)

# Define a resnet block
# modified from https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/models/networks.py
class ResnetBlock(nn.Module):
    def __init__(self, dim, padding_type='reflect', norm_layer=nn.BatchNorm2d, use_dropout=False, use_bias=False):
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, use_dropout, use_bias)

    def build_conv_block(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        conv_block = []
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias), 
                       norm_layer(dim),
                       nn.ReLU(True)]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias),
                       norm_layer(dim)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        out = x + self.conv_block(x)
        return out

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet_Cifar(nn.Module):
    def __init__(self, block, layers, width=1, num_classes=10):
        super(ResNet_Cifar, self).__init__()
        self.inplanes = 16
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3,
                            stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(block, 16 * width, layers[0])
        self.layer2 = self._make_layer(block, 32 * width, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 64 * width, layers[2], stride=2)
        self.avgpool = nn.AvgPool2d(8, stride=1)
        self.fc = nn.Linear(64 * block.expansion * width, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                            kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion)
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


class Generator(nn.Module):
    def __init__(self,
                 min_sigma,
                 max_sigma,
                 gen_input_nc=3,
                 image_nc=3,
                 ):
        super(Generator, self).__init__()
        encoder_lis = [
            # CIFAR:3*32*32
            nn.Conv2d(gen_input_nc, 16, kernel_size=1, stride=1, padding=0, bias=True),
            nn.InstanceNorm2d(16),
            nn.ReLU(),
            # 16*32*32
            nn.Conv2d(16, 32, kernel_size=4, stride=2, padding=(1,1), bias=True),
            nn.InstanceNorm2d(32),
            nn.ReLU(),
            # 32*16*16
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=(1,1), bias=True),
            nn.InstanceNorm2d(64),
            nn.ReLU(),
            # 64*8*8
        ]

        bottle_neck_lis = [ResnetBlock(64),
                       ResnetBlock(64),
                       ResnetBlock(64),
                       ResnetBlock(64),]

        decoder_lis = [
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=(1,1), bias=False),
            nn.InstanceNorm2d(32),
            nn.ReLU(),
            # state size. 32 x 16 x 16
            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=(1,1), bias=False),
            nn.InstanceNorm2d(16),
            nn.ReLU(),
            # state size. 16 x 32 x 32
            nn.ConvTranspose2d(16, image_nc, kernel_size=1, stride=1, padding=0, bias=False),
            # state size. image_nc x 32 x 32
            nn.Sigmoid(),
        ]

        self.encoder = nn.Sequential(*encoder_lis)
        self.bottle_neck = nn.Sequential(*bottle_neck_lis)
        self.decoder = nn.Sequential(*decoder_lis)
        self.min_sigma = min_sigma
        self.max_sigma = max_sigma

    def forward(self, x):
        x = self.encoder(x)
        x = self.bottle_neck(x)
        x = self.decoder(x)
        return self.min_sigma * (1-x) + self.max_sigma * x