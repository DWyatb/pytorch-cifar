import os
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
from models import *

class NumpyDataset(Dataset):
    def __init__(self, x, y, transform=None):
        self.x = x
        self.y = y
        self.transform = transform

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        img = self.x[idx]
        label = int(self.y[idx])
        img = np.array(img)
        if img.ndim == 3 and img.shape[0] == 3 and img.shape[1] in (32, 28, 64):
            img = np.transpose(img, (1, 2, 0))
        if img.ndim == 1 and img.size == 32 * 32 * 3:
            img = img.reshape(32, 32, 3)
        if img.ndim == 2 and (img.shape == (3072, 1) or img.shape == (1, 3072)):
            img = img.reshape(32, 32, 3)
        if img.ndim == 2:
            img = np.stack([img, img, img], axis=-1)
        if img.dtype != np.uint8:
            if img.max() <= 1.0:
                img = (img * 255.0).round().astype(np.uint8)
            else:
                img = np.clip(img, 0, 255).astype(np.uint8)
        from PIL import Image
        img = Image.fromarray(img)
        if self.transform:
            img = self.transform(img)
        return img, label

def load_data(client_id, batch_size=128, return_test_names=False):
    DATA_PATH = "../cifar10_fin_new.npz"
    data = np.load(DATA_PATH, allow_pickle=True)
    if client_id == 1:
        x_train = data["x_client1_key"]
        y_train = data["y_client1_key"].astype(np.int64)
    else:
        x_train = data[f"x_client{client_id}_key"]
        y_train = data[f"y_client{client_id}_key"].astype(np.int64)
    all_keys = list(data.keys())
    test_keys = sorted([k for k in all_keys if k.startswith("x_test")])
    x_tests, y_tests, test_names = [], [], []
    for key in test_keys:
        x_tests.append(data[key])
        if "9" in key and "y_test9" in data:
            y_tests.append(data["y_test9"].astype(np.int64))
        elif "y_test" in data:
            y_tests.append(data["y_test"].astype(np.int64))
        else:
            raise KeyError("No find y_test / y_test9")
        test_names.append(key)
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010)),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010)),
    ])
    trainset = NumpyDataset(x_train, y_train, transform=transform_train)
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=0)
    testloaders = []
    for x_test, y_test in zip(x_tests, y_tests):
        assert len(x_test) == len(y_test), f"Length mismatch: {len(x_test)} vs {len(y_test)}"
        testset = NumpyDataset(x_test, y_test, transform=transform_test)
        testloader = DataLoader(testset, batch_size=100, shuffle=False, num_workers=0)
        testloaders.append(testloader)
    num_examples = {"trainset": len(trainset)}
    for name, y_test in zip(test_names, y_tests):
        num_examples[name] = len(y_test)
    if return_test_names:
        return trainloader, testloaders, test_names
    else:
        return trainloader, testloaders, num_examples
