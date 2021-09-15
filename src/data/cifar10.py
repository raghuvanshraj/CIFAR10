from typing import Callable

import torch.utils.data as data
from torch.utils.data.dataset import T_co
from torchvision import datasets


class CIFAR10(data.Dataset):

    def __init__(
            self,
            train: bool,
            root: str,
            transform: Callable,
    ):
        self.data = datasets.CIFAR10(root=root, train=train, download=True, transform=transform)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index) -> T_co:
        return self.data[index]
