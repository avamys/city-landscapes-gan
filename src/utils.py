import os

import torch
from torchvision.utils import save_image


def get_default_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")


def to_device(data, device):
    if isinstance(data, (list, tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)


def denorm(img_tensors, stats):
    return img_tensors * stats[1][0] + stats[0][0]


def save_samples(generator, sample_dir, index, latent_tensors, stats):
    fake_images = generator(latent_tensors)
    fake_fname = "generated-images-{0:0=4d}.png".format(index)
    save_image(
        denorm(fake_images[:16, :, :, :], stats),
        os.path.join(sample_dir, fake_fname),
        nrow=4,
    )
    print("Saving", fake_fname)
