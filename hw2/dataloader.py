from eval_utils import downstream_validation
import utils
import data_utils
import csv
import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset

class cbow_loader(Dataset):
    def __init__(self, args, # *|CURSOR_MARCADOR|*
    encoded_sentences, lens, vocab_to_index, suggested_padding_len):
        self.context_window = args.context_window
        self.truncate = True
        self.encoded_sentences = encoded_sentences
        self.lens = lens
        self.vocab_to_index = vocab_to_index
        self.suggested_padding_len = suggested_padding_len
        self.dataset = []

    def __getitem__(self, idx):
        return self.dataset[idx]


    def __len__(self):
        return self.len