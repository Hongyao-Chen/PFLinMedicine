import numpy as np
import os
import random
import torchvision.transforms as transforms
import torch.utils.data as data
from utils.dataset_utils import split_data, save_file
from os import path
from PIL import Image
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

# Define the PACS dataset paths
PACS_DOMAINS = ['art_painting', 'cartoon', 'photo', 'sketch']

def load_pacs(base_path, domain):
    print(f"load {domain}")
    domain_path = os.path.join(base_path, domain)
    dataset = ImageFolder(domain_path)
    images = []
    labels = []
    for img_path, label in dataset.imgs:
        img = Image.open(img_path).convert('RGB')
        images.append(np.array(img))
        labels.append(label)
    images = np.array(images)
    labels = np.array(labels)
    return images, labels

class PACS_Dataset(data.Dataset):
    def __init__(self, data, labels, transform=None, target_transform=None):
        super(PACS_Dataset, self).__init__()
        self.data = data
        self.labels = labels
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        img, label = self.data[index], self.labels[index]
        img = Image.fromarray(img)
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            label = self.target_transform(label)
        return img, label

    def __len__(self):
        return len(self.data)

def pacs_dataset_read(base_path, domain):
    images, labels = load_pacs(base_path, domain)
    # Define the transform function
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    # Create train and test data loaders
    dataset = PACS_Dataset(data=images, labels=labels, transform=transform)
    # Split dataset into train and test
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = data.random_split(dataset, [train_size, test_size])
    train_loader = DataLoader(dataset=train_dataset, batch_size=len(train_dataset), shuffle=False)
    test_loader = DataLoader(dataset=test_dataset, batch_size=len(test_dataset), shuffle=False)
    return train_loader, test_loader

random.seed(1)
np.random.seed(1)
data_path = "PACS/"
dir_path = "PACS/"

# Allocate data to users
def generate_dataset(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    # Setup directory for train/test data
    config_path = dir_path + "config.json"
    train_path = dir_path + "train/"
    test_path = dir_path + "test/"

    if not os.path.exists(train_path):
        os.makedirs(train_path)
    if not os.path.exists(test_path):
        os.makedirs(test_path)

    root = data_path + "rawdata"

    # Get PACS data
    if not os.path.exists(root):
        os.makedirs(root)
        # Download PACS dataset
        # You need to manually download the PACS dataset and place it in the rawdata directory
        # PACS dataset can be downloaded from: https://domaingeneralization.github.io/#data
        print("Please download the PACS dataset and place it in the rawdata directory.")
        return

    X, y = [], []
    domains = PACS_DOMAINS
    for d in domains:
        train_loader, test_loader = pacs_dataset_read(root, d)

        for _, tt in enumerate(train_loader):
            train_data, train_label = tt
        for _, tt in enumerate(test_loader):
            test_data, test_label = tt

        dataset_image = []
        dataset_label = []

        dataset_image.extend(train_data.cpu().detach().numpy())
        dataset_image.extend(test_data.cpu().detach().numpy())
        dataset_label.extend(train_label.cpu().detach().numpy())
        dataset_label.extend(test_label.cpu().detach().numpy())

        X.append(np.array(dataset_image))
        y.append(np.array(dataset_label))

    # Number of clients per domain
    num_clients_per_domain = 5  # Since there are 4 domains, 5 clients per domain will result in 20 clients
    num_clients = num_clients_per_domain * len(domains)
    print(f'Number of clients: {num_clients}')

    # Split each domain's data into `num_clients_per_domain` parts
    X_split = []
    y_split = []
    for i in range(len(X)):
        X_split.append(np.array_split(X[i], num_clients_per_domain))
        y_split.append(np.array_split(y[i], num_clients_per_domain))

    # Assign each split to a client
    X_combined = []
    y_combined = []
    for domain_idx in range(len(domains)):
        for client_idx in range(num_clients_per_domain):
            X_combined.append(X_split[domain_idx][client_idx])
            y_combined.append(y_split[domain_idx][client_idx])

    # Calculate statistics for each client
    statistic = [[] for _ in range(num_clients)]
    for client in range(num_clients):
        for i in np.unique(y_combined[client]):
            statistic[client].append((int(i), int(sum(y_combined[client] == i))))

    # Split data into train and test sets for each client
    train_data, test_data = split_data(X_combined, y_combined)
    save_file(config_path, train_path, test_path, train_data, test_data, num_clients, max([len(np.unique(yy)) for yy in y_combined]),
              statistic, None, None, None)

if __name__ == "__main__":
    generate_dataset(dir_path)