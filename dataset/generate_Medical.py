# PFLlib: Personalized Federated Learning Algorithm Library
# Copyright (C) 2021  Jianqing Zhang

# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License along
# with this program; if not, write to the Free Software Foundation, Inc.,
# 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.

import numpy as np
import os
import sys
import random
import torch
import torchvision
import torchvision.transforms as transforms
from matplotlib import pyplot as plt
from medmnist import PathMNIST, OrganSMNIST

from utils.dataset_utils import check, separate_data, split_data, save_file

from medmnist import PathMNIST, OCTMNIST, OrganSMNIST, OrganCMNIST, PneumoniaMNIST, RetinaMNIST
random.seed(1)
np.random.seed(1)
num_clients = 20

# Allocate data to users
def generate_dataset(num_clients, niid, balance, partition, dataname):
    dir_path = dataname+"/"
    rawdata_path = os.path.join(dir_path, "rawdata")
    if not os.path.exists(rawdata_path):
        os.makedirs(rawdata_path)
    config_path = dir_path + "config.json"
    train_path = dir_path + "train/"
    test_path = dir_path + "test/"
    if not os.path.exists(train_path):
        os.makedirs(train_path)
        os.makedirs(test_path)

    if dataname == 'PathMNIST':
        channel = 3
        im_size = (28, 28)
        num_classes = 9
        mean = [0.5, 0.5, 0.5]
        std = [0.5, 0.5, 0.5]
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)])

        # Load raw data (保持与EMNIST相同的处理逻辑)
        trainset = PathMNIST(split="train", download=True, root=rawdata_path, transform=transform)
        testset = PathMNIST(split="test", download=True, root=rawdata_path, transform=transform)


    if dataname == 'OrganSMNIST':
        channel = 1
        im_size = (28, 28)
        num_classes = 11
        mean = [0.5,]
        std = [0.5,]
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)])

        # Load raw data (保持与EMNIST相同的处理逻辑)
        trainset = OrganSMNIST(split="train", download=True, root=rawdata_path, transform=transform)
        testset = OrganSMNIST(split="test", download=True, root=rawdata_path, transform=transform)


    if dataname == 'OCTMNIST':
        channel = 1
        im_size = (28, 28)
        num_classes = 4
        mean = [0.5,]
        std = [0.5,]
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)])

        # Load raw data (保持与EMNIST相同的处理逻辑)
        trainset = OCTMNIST(split="train", download=True, root=rawdata_path, transform=transform)
        testset = OCTMNIST(split="test", download=True, root=rawdata_path, transform=transform)

    if dataname == 'OrganCMNIST224':
        channel = 1
        im_size = (224, 224)
        num_classes = 11
        mean = [0.5,]
        std = [0.5,]
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)])

        # Load raw data (保持与EMNIST相同的处理逻辑)
        trainset = OrganCMNIST(split="train", download=True, root=rawdata_path, transform=transform, size=224)
        testset = OrganCMNIST(split="test", download=True, root=rawdata_path, transform=transform, size=224)
    if dataname == 'PneumoniaMNIST224':
        channel = 1
        im_size = (224, 224)
        num_classes = 2
        mean = [0.5,]
        std = [0.5,]
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)])

        # Load raw data (保持与EMNIST相同的处理逻辑)
        trainset = PneumoniaMNIST(split="train", download=True, root=rawdata_path, transform=transform, size=224)
        testset = PneumoniaMNIST(split="test", download=True, root=rawdata_path, transform=transform, size=224)
    if dataname == 'RetinaMNIST224':
        channel = 3
        num_classes = 5
        im_size = (224, 224)
        mean = [0.5, 0.5, 0.5]
        std = [0.5, 0.5, 0.5]
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)])

        # Load raw data (保持与EMNIST相同的处理逻辑)
        trainset = RetinaMNIST(split="train", download=True, root=rawdata_path, transform=transform, size=224)
        testset = RetinaMNIST(split="test", download=True, root=rawdata_path, transform=transform, size=224)

    trainset.labels = np.array(np.squeeze(trainset.labels).tolist(), dtype='int64')
    testset.labels = np.array(np.squeeze(testset.labels).tolist(), dtype='int64')

    # 使用DataLoader批量加载数据 (与原始代码相同的处理流程)
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=len(trainset.imgs), shuffle=False)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=len(testset.imgs), shuffle=False)

    # 提取数据到numpy数组 (保持与EMNIST相同的变量名和结构)
    for _, train_data in enumerate(trainloader, 0):
        trainset.imgs, trainset.labels = train_data
    for _, test_data in enumerate(testloader, 0):
        testset.imgs, testset.labels = test_data

    dataset_image = []
    dataset_label = []
    dataset_image.extend(trainset.imgs.cpu().detach().numpy())
    dataset_image.extend(testset.imgs.cpu().detach().numpy())
    dataset_label.extend(trainset.labels.cpu().detach().numpy())
    dataset_label.extend(testset.labels.cpu().detach().numpy())
    dataset_image = np.array(dataset_image)
    dataset_label = np.array(dataset_label)

    # 后续保持与原始generate_dataset()相同的分割逻辑
    X, y, statistic = separate_data((dataset_image, dataset_label), num_clients, num_classes,
                                    niid, balance, partition, class_per_client=2)

    train_data, test_data = split_data(X, y)
    save_file(config_path, train_path, test_path, train_data, test_data, num_clients, num_classes,
              statistic, niid, balance, partition)

if __name__ == "__main__":
    generate_dataset(num_clients, True, '-', 'dir',"PathMNIST")