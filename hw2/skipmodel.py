
# IMPLEMENT YOUR MODEL CLASS HERE

import numpy as np
import torch
import torch.nn.functional as F

device = torch.device('cpu')

class skip(torch.nn.Module):
    def __init__(self, args):
        super(skip, self).__init__()
        self.vocab_size = args.vocab_size
        self.context_window = args.context_window
        self.padding_idx = 0
        self.embedding_dim = 128
        self.embedding = torch.nn.Embedding(self.vocab_size, self.embedding_dim, padding_idx=self.padding_idx)
        self.linear = torch.nn.Linear(self.embedding_dim, self.vocab_size)
        self.verbose = False
        self.args = args
    def forward(self, x):
        if self.verbose:
            print("x size", x.size())
        embed = self.embedding(x)
        if self.verbose:
            print("embed size", embed.size())
        if self.verbose:
            print("embed size", embed.size())
        out = self.linear(embed)
        if self.verbose:
            print("out size", out.size())
        return out
        
        
