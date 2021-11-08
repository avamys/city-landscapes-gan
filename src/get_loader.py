import torch
import tarfile
import random
from PIL import Image
from torch.utils.data import DataLoader, Dataset
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
