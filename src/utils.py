import os
import torch
from torchvision.utils import save_image
from torchvision.utils import make_grid
import matplotlib.pyplot as plt


def get_default_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')
            
def to_device(data, device):
    if isinstance(data, (list,tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)

def denorm(img_tensors, stats):
    return img_tensors * img_tensors * stats[1][0] + stats[0][0]

def save_samples(generator, sample_dir, index, latent_tensors, stats, show=True):
    fake_images = generator(latent_tensors)
    fake_fname = 'generated-images-{0:0=4d}.png'.format(index)
    save_image(denorm(fake_images, stats), os.path.join(sample_dir, fake_fname), nrow=2)
    print('Saving', fake_fname)
    if show:
        fig, ax = plt.subplots(figsize=(4, 4))
        ax.set_xticks([]); ax.set_yticks([])
        ax.imshow(make_grid(fake_images.cpu().detach(), nrow=2).permute(1, 2, 0))
