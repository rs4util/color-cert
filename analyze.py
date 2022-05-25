import numpy as np
import matplotlib.pyplot as plt

def sigma_heatmap(epoch, step, imgs, sigma, entropy):
    '''
        Input: 
            imgs: NumPy array of images (image_num, height, width, channel)
            entropy: pixel entropy of the imgs (image_num, height, width, 1)
            sigma: sigmas of imgs, in shape of (image_num, height, width, channel), NumPy array.
        Output: 
            heatmaps of sigma and the corresponding images
    '''
    sigma = (sigma-sigma.min())/(sigma.max() - sigma.min() + 1e-9)
    entropy = (entropy-entropy.min())/(entropy.max() - entropy.min() + 1e-9)
    fig, axes = plt.subplots(nrows=3, ncols=10)
    for i in range(min(10, len(imgs))):
        im = axes[0][i].imshow(imgs[i])
        im2 = axes[1][i].imshow(np.squeeze(entropy[i], axis=-1), cmap='gray')
        im3 = axes[2][i].imshow(sigma[i])
        axes[0][i].axis('off')
        axes[1][i].axis('off')
        axes[2][i].axis('off')
    axes[0][5].set_title('Image')
    axes[1][5].set_title('Entropy')
    axes[2][5].set_title('Simga')
    plt.savefig(f'sigma_log/epoch_{epoch}_step{step}.png')
    plt.close()

