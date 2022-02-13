import torch.nn as nn


class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()

        self.discriminator = nn.Sequential(
            # in: 3 x 128 x 128
            nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            # out: 64 x 64 x 64
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            # out: 128 x 32 x 32
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            # out: 256 x 16 x 16
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            # out: 512 x 8 x 8
            nn.Conv2d(512, 1, kernel_size=8, stride=1, padding=0, bias=False),
            nn.Flatten(),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.discriminator(x)


class Generator(nn.Module):
    def __init__(self, latent_size):
        super().__init__()
        self.latent_size = latent_size

        self.generator = nn.Sequential(
            # in: latent_size x 1 x 1
            nn.ConvTranspose2d(
                latent_size, 512, kernel_size=8, stride=1, padding=0, bias=False
            ),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            # out: 256 x 8 x 8
            nn.ConvTranspose2d(
                512, 256, kernel_size=4, stride=2, padding=1, bias=False
            ),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            # out: 128 x 16 x 16
            nn.ConvTranspose2d(
                256, 128, kernel_size=4, stride=2, padding=1, bias=False
            ),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            # out: 64 x 32 x 32
            nn.ConvTranspose2d(
                128, 64, kernel_size=4, stride=2, padding=1, bias=False
            ),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            # out: 64 x 32 x 32
            nn.ConvTranspose2d(
                64, 3, kernel_size=4, stride=2, padding=1, bias=False
            ),
            nn.Tanh()
            # out: 3 x 64 x 64
        )

    def forward(self, x):
        return self.generator(x)
