import os
import tarfile

import torchvision.transforms as T
from PIL import Image
from torch.utils.data import ConcatDataset, DataLoader, Dataset

from utils import to_device


class DeviceDataLoader:
    def __init__(self, loader, device):
        self.loader = loader
        self.device = device

    def __iter__(self):
        for img in self.loader:
            yield to_device(img, self.device)

    def __len__(self):
        return len(self.loader)


class TarfileDataset(Dataset):
    def __init__(self, tar_path, transform=None):
        self.tar = tarfile.open(tar_path)
        self.tarlist = self.tar.getmembers()
        self.transform = transform

    def __len__(self):
        return len(self.tarlist)

    def __getitem__(self, index):
        f = self.tar.extractfile(self.tarlist[index])
        img = Image.open(f)

        if self.transform is not None:
            img = self.transform(img)

        return img


class ImageDataset(Dataset):
    def __init__(self, path, is_split=False, transform=None):
        self.path = path
        self.images = os.listdir(path)
        self.transform = transform

        if is_split:
            self.images = []
            for imdir in os.listdir(path):
                impaths = [f"{imdir}/{x}" for x in os.listdir(f"{path}/{imdir}")]
                self.images.extend(impaths)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img = Image.open(f"{self.path}/{self.images[index]}").convert("RGB")

        if self.transform is not None:
            img = self.transform(img)

        return img


def get_loader(
    root_folder,
    transform,
    batch_size=256,
    num_workers=8,
    img_size=128,
    shuffle=True,
    pin_memory=True,
    is_split=False,
    augment=None,
):

    if augment:
        seq_dataset = [
            ImageDataset(root_folder, is_split=is_split, transform=transform)
        ]
        for _ in range(augment):
            aug_transform = T.Compose(
                [T.RandomResizedCrop(size=img_size, scale=(0.5, 1.0)), transform]
            )
            aug_dataset = ImageDataset(
                root_folder, is_split=is_split, transform=aug_transform
            )
            seq_dataset.append(aug_dataset)
        dataset = ConcatDataset(seq_dataset)
    else:
        dataset = ImageDataset(root_folder, is_split=is_split, transform=transform)

    loader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=shuffle,
        pin_memory=pin_memory,
        drop_last=True,
    )

    return loader, dataset
