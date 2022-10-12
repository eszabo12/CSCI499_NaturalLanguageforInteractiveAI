from eval_utils import downstream_validation
import utils
import data_utils
import csv
import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset

class cbow_data(Dataset):
    def __init__(self, args, dataset):
        self.dataset = dataset
        self.len = len(dataset)

    def __getitem__(self, idx):
        context, output = self.dataset[idx]
        return torch.tensor(context), torch.tensor(output)

    def __len__(self):
        return self.len