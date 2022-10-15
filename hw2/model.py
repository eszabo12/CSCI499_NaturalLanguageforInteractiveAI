
# IMPLEMENT YOUR MODEL CLASS HERE

import numpy as np
import torch
import torch.nn.functional as F

device = torch.device('cpu')

class cbow(torch.nn.Module):
    def __init__(self, args):
        super(cbow, self).__init__()
        self.vocab_size = args.vocab_size
        context_window = args.context_window
        self.padding_idx = 0
        self.embedding_dim = 128
        self.embedding = torch.nn.Embedding(self.vocab_size, self.embedding_dim, padding_idx=self.padding_idx)
        self.linear = torch.nn.Linear(self.embedding_dim, self.vocab_size)
        self.verbose = False
        self.args = args
    def forward(self, x):
        holder = torch.zeros(self.args.batch_size, self.embedding_dim)
        if self.verbose:
            print("x size", x.size())
        for idx, sample in enumerate(x):
            sum = torch.zeros(self.embedding_dim)
            if self.verbose:
                print("sum", sum.size())
                print("sample", sample)
            for word in sample:
                if self.verbose:
                    print("word", word)
                    print("word size", word.size())
                embed = self.embedding(word)
                if self.verbose:
                    print("embed size", embed.size())
                sum = torch.add(sum, embed)
            holder[idx] = sum
        # if self.verbose:
        # print("holder size", holder.size())
        out = self.linear(holder)
        if self.verbose:
            print("out size", out.size())
        return out
        
        
