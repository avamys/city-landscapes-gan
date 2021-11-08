import torch
import pandas as pd
import torchvision.transforms as T

from model import Discriminator, Generator
from get_loader import DeviceDataLoader, get_loader
from utils import get_default_device, to_device
from training import fit


if __name__ == "__main__":
    lr = 0.0002
    epochs = 50
    image_folder = "train"
    image_size = 64
    batch_size = 64
    num_workers = 2
    stats = (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)
    latent_size = 128
    device = get_default_device()
    transform = T.Compose([
        T.Resize(image_size),
        T.CenterCrop(image_size),
        T.ToTensor(),
        T.Normalize(*stats)
    ])

    train_loader, dataset = get_loader(
        root_folder=image_folder,
        transform=transform,
        batch_size=batch_size,
        num_workers=num_workers,
    )

    train = DeviceDataLoader(train_loader, device)

    discriminator = Discriminator()
    generator = Generator(latent_size)

    torch.cuda.empty_cache()

    discriminator = to_device(discriminator, device)
    generator = to_device(generator, device)

    history = fit(discriminator, generator, epochs, lr, train, stats)
    history = pd.DataFrame(
        data=history, 
        columns=['losses_g', 'losses_d', 'real_scores', 'fake_scores'])
    history.to_csv('history.csv', index=False)
