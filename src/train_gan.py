import click
import pandas as pd
import torch
import torchvision.transforms as T
import yaml

from get_loader import DeviceDataLoader, get_loader
from model import Discriminator, Generator
from training import fit
from utils import get_default_device, to_device


@click.command()
@click.argument("config_file", type=click.Path())
@click.argument("image_folder", type=str)
def main(config_file, image_folder):
    with open(config_file) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    lr = config["lr"]
    epochs = config["epochs"]
    image_folder = image_folder
    image_size = config["image_size"]
    batch_size = config["batch_size"]
    num_workers = config["num_workers"]
    stats = config["stats"]
    latent_size = config["latent_size"]
    augment = config["augment"]
    load = config["load"]
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
        generator.load_state_dict(
            torch.load(f"model_backups/{config['generator']}.pth")
        )
        discriminator.load_state_dict(
            torch.load(f"model_backups/{config['discriminator']}.pth")
        )

    torch.cuda.empty_cache()

    discriminator = to_device(discriminator, device)
    generator = to_device(generator, device)

    history = fit(
        discriminator,
        generator,
        epochs,
        lr,
        train,
        stats,
        latent_size,
        batch_size,
        device,
    )
    history_log = pd.DataFrame(
        data=history, columns=["losses_g", "losses_d", "real_scores", "fake_scores"]
    )
    history_log.to_csv("history.csv", index=False)


if __name__ == "__main__":
    main()
