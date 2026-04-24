# PFLlib: Personalized Federated Learning Algorithm Library
# Copyright (C) 2021  Jianqing Zhang

import numpy as np
import os
import sys
import random
import pandas as pd
from torch.utils.data import Dataset
from utils.dataset_utils import check, separate_data, split_data, save_file
from utils.language_utils import tokenizer

random.seed(1)
np.random.seed(1)
num_clients = 20
max_len = 200
max_tokens = 32000
dir_path = "AGNews/"
samples_per_class = 5000  # 每个类别只使用5000条数据


class AGNewsLocalDataset(Dataset):
    def __init__(self, csv_path, limit_per_class=None):
        # 读取本地CSV文件
        self.data = pd.read_csv(csv_path, header=None, names=["label", "title", "description"])

        # 如果指定了limit_per_class，则对每个类别进行采样
        if limit_per_class is not None:
            self.data = self.data.groupby('label').apply(
                lambda x: x.sample(min(len(x), limit_per_class), random_state=1)
            ).reset_index(drop=True)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        # 返回格式与原始AG_NEWS数据集一致：(label, text)
        return row["label"], f"{row['title']} {row['description']}"


# Allocate data to users
def generate_dataset(dir_path, num_clients, niid, balance, partition):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    # Setup directory for train/test data
    config_path = dir_path + "config.json"
    train_path = dir_path + "train/"
    test_path = dir_path + "test/"

    if check(config_path, train_path, test_path, num_clients, niid, balance, partition):
        return

    # 修改点1：使用本地文件代替在线下载
    raw_data_path = os.path.join(dir_path, "rawdata")
    if not os.path.exists(raw_data_path):
        os.makedirs(raw_data_path)

    # 假设你已经手动下载了train.csv和test.csv到rawdata目录
    train_file = os.path.join(raw_data_path, "train.csv")
    test_file = os.path.join(raw_data_path, "test.csv")

    # 检查文件是否存在
    if not os.path.exists(train_file) or not os.path.exists(test_file):
        raise FileNotFoundError(
            f"请手动下载AG News数据集并保存为 {train_file} 和 {test_file}\n"
            "下载链接:\n"
            "train.csv: https://raw.githubusercontent.com/mhjabreel/CharCnn_Keras/master/data/ag_news_csv/train.csv\n"
            "test.csv: https://raw.githubusercontent.com/mhjabreel/CharCnn_Keras/master/data/ag_news_csv/test.csv"
        )

    # 修改点2：使用本地数据集类并限制每个类别的样本数
    trainset = AGNewsLocalDataset(train_file, limit_per_class=samples_per_class)
    testset = AGNewsLocalDataset(test_file, limit_per_class=1000)

    trainlabel, traintext = list(zip(*trainset))
    testlabel, testtext = list(zip(*testset))

    dataset_text = []
    dataset_label = []

    dataset_text.extend(traintext)
    dataset_text.extend(testtext)
    dataset_label.extend(trainlabel)
    dataset_label.extend(testlabel)

    num_classes = len(set(dataset_label))
    print(f'Number of classes: {num_classes}')
    print(f'Total samples per class: {samples_per_class}')
    print(f'Total training samples: {len(trainset)}')
    print(f'Total testing samples: {len(testset)}')

    vocab, text_list = tokenizer(dataset_text, max_len, max_tokens)
    label_pipeline = lambda x: int(x) - 1
    label_list = [label_pipeline(l) for l in dataset_label]

    text_lens = [len(text) for text in text_list]
    text_list = [(text, l) for text, l in zip(text_list, text_lens)]

    text_list = np.array(text_list, dtype=object)
    label_list = np.array(label_list)

    X, y, statistic = separate_data((text_list, label_list), num_clients, num_classes, niid, balance, partition)
    train_data, test_data = split_data(X, y)
    save_file(config_path, train_path, test_path, train_data, test_data, num_clients, num_classes,
              statistic, niid, balance, partition)

    print("The size of vocabulary:", len(vocab))


if __name__ == "__main__":
    generate_dataset(dir_path, num_clients, True, '-', 'dir')