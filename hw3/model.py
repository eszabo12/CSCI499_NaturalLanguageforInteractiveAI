# IMPLEMENT YOUR MODEL CLASS HERE

import torch.nn as nn
import random
import torch
import torch.nn.functional as F


verbose = True
class Encoder(nn.Module):
    """
    Encode a sequence of tokens. Run the input sequence
    through any recurrent model and output a hidden representation.
    TODO: edit the forward pass arguments to suit your needs
    """

    def __init__(self, args):
        super(Encoder, self).__init__()
        self.embedding = nn.Embedding(args.vocab_size, args.embedding_dim)
        self.gru = nn.GRU(args.embedding_dim, args.embedding_dim)
        self.batch_size = args.batch_size
        self.args = args
    def forward(self, x, hidden):
        if hidden == None:
            hidden = torch.zeros(1, self.args.instruction_length, self.args.embedding_dim)
        # print("x, hidden", x.size(), hidden.size())
        embedded = self.embedding(x)
        # print("embedded", embedded.size())
        output, hidden = self.gru(embedded, hidden)
        # print("output, hidden", output.size(), hidden.size())
        return output, hidden


class Decoder(nn.Module):
    """
    Conditional recurrent decoder. Iteratively generates the next
    token given the context vector from the encoder and ground truth
    labels using teacher forcing.
    TODO: edit the forward pass arguments to suit your needs
    """

    def __init__(self, args):
        super(Decoder, self).__init__()
        self.embedding = nn.Embedding(args.output_size, args.embedding_dim)
        self.attention = nn.Linear(args.embedding_dim, args.embedding_dim)
        self.gru = nn.GRU(args.embedding_dim, 1)
        # self.out = nn.Linear(args.embedding_dim, args.output_size)
        self.out2 = nn.Linear(args.instruction_length, args.output_size)
        self.args = args
        self.ratio = args.teacher_ratio
        self.num_actions = 9 #added stop

    def forward(self, decoder_input, decoder_hidden, encoder_outputs):
        # print("decoder_input:", decoder_input)
        # print("hidden:", decoder_hidden.size())
        # print("encoder_outputs:", encoder_outputs.size())
        embedded = self.embedding(decoder_input)
        if verbose:
            print("embedded:", embedded.size())
        output = torch.cat((embedded, encoder_outputs), dim=1)
        print("output:", output.size())
        output = self.attention(encoder_outputs)
        print("output:", output.size())
        # if verbose:
            # print("output:", output.size())
        output = F.relu(output)
        print("output:", output.size())
        # if verbose:
            # print("output:", output.size())
        output, decoder_hidden = self.gru(output, decoder_hidden)
        # if verbose:
        print("output2:", output.size())
        # output = F.log_softmax(self.out(output[0]), dim=1)
        # output = self.out(output)
        # print("output:", output.size())
        output = self.out2(F.relu(output.flatten(1,2)))
        if verbose:
            print("output3:", output.size())
        return output, decoder_hidden

class EncoderDecoder(nn.Module):
    """
    Wrapper class over the Encoder and Decoder.
    TODO: edit the forward pass arguments to suit your needs
    """

    def __init__(self, args, maps, act_maps):
        super(EncoderDecoder, self).__init__()
        self.vocab_to_index, self.index_to_vocab, self.num_pad, self.max_len = maps
        self.args = args
        self.ratio = args.teacher_ratio
        self.encoder = Encoder(args)
        self.decoder = Decoder(args)
        self.actions_to_index, self.index_to_actions, self.targets_to_index, self.index_to_targets = act_maps

    def forward(self, x, y):
        teacher_forcing = random.random() > self.ratio
        if verbose:
            print("teacher forcing:", teacher_forcing)
        print("x. y", x.size(), y.size()) #x. y r.Size([51, 684]) torch.Size([51, 12, 2])
        batch_size = 51
        hidden = None
        inputs, outputs = x.int(), y.int()
        encoding, hidden = self.encoder(inputs.to(torch.int64), hidden)
        d_input = None
        decoder_outputs = torch.empty(batch_size, self.args.targets_length, self.args.output_size)
        print("Decoder outputs size,", decoder_outputs.size())
        d_output = torch.zeros(self.args.output_size).to(torch.int64)
        for i in range(self.args.targets_length):
            if teacher_forcing and i != 0:
                d_input = outputs[:,i-1] # need to concatenate
            else:
                if i == 0:
                     d_input = torch.zeros(batch_size, 2).to(torch.int64)
                else:
                    #returns tuple of values, indices
                    print("student forced")
                    act = d_output[:self.args.num_actions]
                    tar = d_output[self.args.num_actions:]
                    a, v = act.topk(1)
                    t, v = tar.topk(1)
                    d_input = torch.Tensor([a, t]).to(torch.int64)
                    # print("D input teacher", d_input)
            print("d_input, ,hidden, encoding:", d_input.size(), hidden.size(), encoding.size())
            d_output, hidden = self.decoder(d_input, hidden, encoding)
            print("d outputs size", decoder_outputs.size())
            decoder_outputs[:, i] = d_output
            # if i >= len(outputs) - 1:
            #     break
        tensor_output = torch.Tensor(decoder_outputs).to(torch.int64)
        return tensor_output