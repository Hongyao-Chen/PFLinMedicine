import numpy as np
import os
import random

import torch
import torchvision.transforms as transforms
import torch.utils.data as data
from utils.dataset_utils import split_data, save_file
from PIL import Image
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

random.seed(1)
np.random.seed(1)

DATA_PATH = "MIDOGpp/"
DIR_PATH = "MIDOGpp/"
RAW_PATH = DATA_PATH

TUMOR_TYPES = [
    'canine_cutaneous_mast_cell_tumor',
    'canine_lung_cancer',
    'canine_lymphosarcoma',
    'canine_soft_tissue_sarcoma',
    'human_breast_cancer',
    'human_melanoma',
    'human_neuroendocrine_tumor'
]


class MIDOGpp_Dataset(data.Dataset):
    def __init__(self, data, labels, transform=None):
        self.data, self.labels = data, labels
        self.transform = transform

    def __getitem__(self, idx):
        img, lbl = self.data[idx], self.labels[idx]
        img = Image.fromarray(img)
        if self.transform:
            img = self.transform(img)
        return img, lbl

    def __len__(self):
        return len(self.data)



def load_tumor_type(base_path, tumor_type):
    print(f"load {tumor_type}")
    folder = os.path.join(base_path, tumor_type)
    dataset = ImageFolder(folder)
    images, labels = [], []
    for img_path, label in dataset.imgs:
        img = Image.open(img_path).convert('RGB')
        images.append(np.array(img))
        labels.append(label)  # 0 或 1
    # print(labels)
    return np.array(images), np.array(labels)



def tumor_dataset_read(base_path, tumor_type):
    images, labels = load_tumor_type(base_path, tumor_type)
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])
    dataset = MIDOGpp_Dataset(images, labels, transform=transform)
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_set, test_set = data.random_split(dataset, [train_size, test_size])
    train_loader = DataLoader(train_set, batch_size=len(train_set), shuffle=False)
    test_loader = DataLoader(test_set, batch_size=len(test_set), shuffle=False)
    return train_loader, test_loader



def generate_dataset(dir_path):
    os.makedirs(dir_path, exist_ok=True)
    config_path = os.path.join(dir_path, "config.json")
    train_path = os.path.join(dir_path, "train")
    test_path = os.path.join(dir_path, "test")
    os.makedirs(train_path, exist_ok=True)
    os.makedirs(test_path, exist_ok=True)


    X, y = [], []
    for client_id, tumor_type in enumerate(TUMOR_TYPES):
        train_loader, test_loader = tumor_dataset_read(RAW_PATH, tumor_type)
        train_data, train_label = next(iter(train_loader))
        test_data, test_label = next(iter(test_loader))

        all_data = torch.cat([train_data, test_data]).cpu().numpy()
        all_label = torch.cat([train_label, test_label]).cpu().numpy()
        X.append(all_data)
        y.append(all_label)


    train_data, test_data = split_data(X, y)
    statistic = []
    for i in range(len(y)):
        uniq, cnt = np.unique(y[i], return_counts=True)
        statistic.append([(int(u), int(c)) for u, c in zip(uniq, cnt)])
    # print(statistic)
    save_file(config_path, train_path, test_path,
              train_data, test_data,
              num_clients=len(TUMOR_TYPES),
              num_classes=2,
              statistic=statistic,
              niid=False, balance=None, partition=None)


if __name__ == "__main__":
    generate_dataset(DIR_PATH)
