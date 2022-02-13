import click
import pandas as pd
import torch
import torchvision.transforms as T

from get_loader import DeviceDataLoader, get_loader
from model import Discriminator, Generator
from training import fit
from utils import get_default_device, to_device

@click.command()
@click.argument('image_folder', type=str)
def main(image_folder):
    lr = 0.0002
    epochs = 100
    image_folder = image_folder
    image_size = 128
    batch_size = 256
    num_workers = 2
    stats = (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)
    latent_size = 128
    augment = 6
    load = False
    device = get_default_device()
    transform = T.Compose(
        [
            T.Resize(image_size),
            T.CenterCrop(image_size),
            T.ToTensor(),
            T.Normalize(*stats),
        ]
    )

    train_loader, dataset = get_loader(
        root_folder=image_folder,
        transform=transform,
        batch_size=batch_size,
        num_workers=num_workers,
        augment=augment,
    )

    train = DeviceDataLoader(train_loader, device)

    discriminator = Discriminator()
    generator = Generator(latent_size)

    if load:
        generator.load_state_dict(torch.load("model_backups/G.pth"))
        discriminator.load_state_dict(torch.load("model_backups/D.pth"))

    torch.cuda.empty_cache()

    discriminator = to_device(discriminator, device)
    generator = to_device(generator, device)

    history = fit(discriminator, generator, epochs, lr, train, stats, 
                  latent_size, batch_size, device)
    history_log = pd.DataFrame(
        data=history, columns=["losses_g", "losses_d", "real_scores", "fake_scores"]
    )
    history_log.to_csv("history.csv", index=False)


if __name__ == "__main__":
    main()
