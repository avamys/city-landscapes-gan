import os

import click
import torch
from torchvision.utils import save_image
from tqdm import tqdm

from model import Generator
from utils import denorm


@click.command()
@click.argument("image_folder", type=str)
def main(image_folder):
    latent_size = 128
    stats = (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)
    N = 100
    generator = Generator(latent_size)
    generator.load_state_dict(torch.load("model_backups/G.pth"))
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
