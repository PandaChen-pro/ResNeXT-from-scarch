import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import pandas as pd
import os
from torchvision import transforms
from torchvision.datasets import ImageFolder

torch.manual_seed(3407)

def load_dataset(train_dataset_path, val_dataset_path, batch_size=32, is_train_shuffle=True, is_val_shuffle=False):
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    train_dataset = ImageFolder(train_dataset_path, transform=train_transform)
    val_dataset = ImageFolder(val_dataset_path, transform=val_transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=is_train_shuffle)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=is_val_shuffle)

    return train_loader, val_loader

if __name__ == "__main__":
    train_dataset_path = "/home/code/experiment/modal/exp-1/datasets/train"
    val_dataset_path = "/home/code/experiment/modal/exp-1/datasets/val"
    batch_size = 32
    is_train_shuffle = True
    is_val_shuffle = False
    train_loader, val_loader = load_dataset(train_dataset_path, val_dataset_path, batch_size, is_train_shuffle, is_val_shuffle)
    print(len(train_loader), len(val_loader))