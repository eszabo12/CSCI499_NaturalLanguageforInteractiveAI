# IMPLEMENT YOUR MODEL CLASS HERE

import torch.nn as nn
import random
import torch
import torch.nn.functional as F
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
            hidden = torch.zeros(1, self.args.embedding_dim, requires_grad=True)
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
        self.out2 = nn.Linear(args.instruction_length, args.output_size)
        self.args = args
        self.ratio = args.teacher_ratio
        self.num_actions = 9 #added stop

    def forward(self, decoder_input, decoder_hidden, encoder_outputs):
        print("decoder_input:", decoder_input.size())
        print("hidden:", decoder_hidden.size())
        print("encoder_outputs:", encoder_outputs.size())
        embedded = self.embedding(decoder_input)
        # if verbose:
        print("embedded:", embedded.size())
        # embedded = embedded.flatten(1, 2)
        # print("embedded:", embedded.size())
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
        # output = self.out2(F.relu(output.flatten(1,2)))
        output = self.out2(F.relu(output.squeeze()))

        # if verbose:
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
        self.encoder = Encoder(args).to(device)
        self.decoder = Decoder(args).to(device)
        self.actions_to_index, self.index_to_actions, self.targets_to_index, self.index_to_targets = act_maps
