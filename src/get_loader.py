import torch
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from utils import to_device


class DeviceDataLoader():
    def __init__(self, loader, device):
        self.loader = loader
        self.device = device
        
    def __iter__(self):
        for img in self.loader: 
            yield to_device(img, self.device)

    def __len__(self):
        return len(self.loader)


def get_loader(
    root_folder,
    transform,
    batch_size=128,
    num_workers=8,
    img_size=64,
    shuffle=True,
    pin_memory=True,
):

    dataset = ImageFolder(root=root_folder, transform=transform)

    loader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=shuffle,
        pin_memory=pin_memory,
        drop_last=True
    )

    return loader, dataset
