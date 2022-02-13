import os

import click
import torch
import yaml
from torchvision.utils import save_image
from tqdm import tqdm

from model import Generator
from utils import denorm


@click.command()
@click.argument("config_file", type=click.Path())
@click.argument("image_folder", type=str)
@click.argument("n", type=int)
def main(config_file, image_folder, n):
    with open(config_file) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    latent_size = config["latent_size"]
    stats = config["stats"]
    N = n
    generator = Generator(latent_size)
    generator.load_state_dict(torch.load(f"model_backups/{config['generator']}.pth"))
    generator.eval()

    latent = torch.randn(N, latent_size, 1, 1)
    images = generator(latent)
    for i in tqdm(range(N)):
        save_image(
            denorm(images[i, :, :, :], stats),
            os.path.join(image_folder, f"image_{i}.png"),
        )


if __name__ == "__main__":
    main()
