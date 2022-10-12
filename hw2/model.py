
# IMPLEMENT YOUR MODEL CLASS HERE

import numpy as np
import torch
import torch.nn.functional as F

device = torch.device('cpu')

class cbow:
    def __init__(args):
        self.vocab_size = args.vocab_size
        context_window = args.context_window
        self.padding_idx = 0
        self.embedding_dim = 128
        self.embedding = torch.nn.Embedding(vocab_size, embedding_dim, padding_idx=self.padding_idx)
        self.linear = torch.nn.Linear(self.embedding_dim, self.vocab_size)
    def forward(self, x):
        embeds = self.embedding(x)