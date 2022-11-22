import tqdm
import torch
from torch.utils.data import DataLoader, Dataset

import argparse
from sklearn.metrics import accuracy_score
import json 
from model import Encoder, Decoder, EncoderDecoder
import random
import einops
import sys

EPSILON = sys.float_info.epsilon
from utils import (
    get_device,
    preprocess_string,
    build_tokenizer_table,
    build_output_tables,
    prefix_match
)
class AlfredData(Dataset):
    #inputs a python dictionary { "dialogue": ["action", "target"]}
    #input size must be <= 57
    def __init__(self, args, train):
        self.actions_to_index, self.index_to_actions, self.targets_to_index, self.index_to_targets = build_output_tables(train)
        self.action_maps = tuple((self.actions_to_index, self.index_to_actions, self.targets_to_index, self.index_to_targets))
        self.vocab_to_index, self.index_to_vocab, self.num_pad, self.max_len = build_tokenizer_table(train, args.vocab_size)
        self.vocab_maps = tuple((self.vocab_to_index, self.index_to_vocab, self.num_pad, self.max_len))
        self.args = args
        self.input_size = args.input_size
        idx = 0
        self.data = []
        ep_length = -1
        max_instr = 0
        for episode_list in train:
            instructions = []
            # print("len episode list", len(episode_list))
            episode_outputs = torch.zeros((args.targets_length, 2))
            j = 0
            instr_len = 0
            for instruction in episode_list:
                inst, (a, t) = instruction
                inst = self.process_words([self.vocab_to_index.get(i, 3) for i in preprocess_string(inst).split()])
                instr_len += len(inst)
                instructions += inst
                episode_outputs[j] = torch.Tensor( [self.actions_to_index[a], self.targets_to_index[t]])
                j += 1
            max_instr = max(max_instr, instr_len)
            episode_outputs[j] =  torch.Tensor( [self.actions_to_index["Stop"], self.targets_to_index["Stop"]])
            instructions = torch.Tensor(instructions + [0]*(args.instruction_length - len(instructions))) #padding
            self.data.append(tuple((instructions, episode_outputs)))
            idx += 1
        print("max instr length", max_instr)
        self.size = idx
        self.num_actions = len(self.index_to_actions)
        self.num_targets = len(self.index_to_targets)

    def process_words(self, instruction):
        #subtract 2 for start and end tokens
        instruction = [1] + instruction + [2]
        num_pad = self.input_size - 2 - len(instruction)
        if num_pad > 0:
            instruction += [0] * num_pad
        #truncates the input to input size
        return instruction[:self.input_size]
    #returns lists corresponding to one episode
    def __getitem__(self, idx):
        inputs, outputs = self.data[idx]
        return inputs, outputs
    def __len__(self):
        return self.size
    def get_maps(self):
        return self.vocab_maps
    def get_act_maps(self):
        return self.action_maps
        
def setup_dataloader(args):
    """
    return:
        - train_loader: torch.utils.data.Dataloader
        - val_loader: torch.utils.data.Dataloader
    """
    # ================== TODO: CODE HERE ================== #
    # Task: Load the training data from provided json file.
    # Perform some preprocessing to tokenize the natural
    # language instructions and labels. Split the data into
    # train set and validataion set and create respective
    # dataloaders.

    # Hint: use the helper functions provided in utils.py
    # ===================================================== #
    whole_data = None
    with open(args.in_data_fn) as f:
        whole_data = json.load(f)
    train = whole_data["train"] #there is a redundant dimension
    train_dataset = AlfredData(args, train)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size)
    # torch.save(train_loader, 'Dataloaders/train_loader.pt')
    maps = train_dataset.get_maps()
    act_maps = train_dataset.get_act_maps()
    # torch.save(map, 'Dataloaders/maps')
    test = whole_data["valid_seen"] #there is a redundant dimension
    test_dataset = AlfredData(args, test)
    val_loader = DataLoader(test_dataset, batch_size=args.batch_size)
    return train_loader, val_loader, maps, act_maps


def setup_model(args, maps, act_maps):
    """
    return:
        - model: YourOwnModelClass
    """
    # ================== TODO: CODE HERE ================== #
    # Task: Initialize your model. Your model should be an
    # an encoder-decoder architecture that encoders the
    # input sentence into a context vector. The decoder should
    # take as input this context vector and autoregressively
    # decode the target sentence. You can define a max length
    # parameter to stop decoding after a certain length.

    # For some additional guidance, you can separate your model
    # into an encoder class and a decoder class.
    # The encoder class forward pass will simply run the input
    # sequence through some recurrent model.
    # The decoder class you will need to implement a teacher
    # forcing mechanism in the forward pass such that instead
    # of feeding the model prediction into the recurrent model,
    # you will give the embedding of the target token.
    # ===================================================== #
    model = EncoderDecoder(args, maps, act_maps)
    return model


def setup_optimizer(args, model):
    """
    return:
        - criterion: loss_fn
        - optimizer: torch.optim
    """
    # ================== TODO: CODE HERE ================== #
    # Task: Initialize the loss function for action predictions
    # and target predictions. Also initialize your optimizer.
    # ===================================================== #
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters())

    return criterion, optimizer


def train_epoch(
    args,
    model,
    loader,
    optimizer,
    criterion,
    device,
    training=True,
):
    """
    # TODO: implement function for greedy decoding.
    # This function should input the instruction sentence
    # and autoregressively predict the target label by selecting
    # the token with the highest probability at each step.
    # Note this is slightly different from the forward pass of
    # your decoder because you want to pick the token
    # with the highest probability instead of using the
    # teacher-forced token.

    # e.g. Input: "Walk straight, turn left to the counter. Put the knife on the table."
    # Output: [(GoToLocation, diningtable), (PutObject, diningtable)]
    # Also write some code to compute the accuracy of your
    # predictions against the ground truth.
    """

    epoch_loss = 0.0
    epoch_acc = 0.0

    # iterate over each batch in the dataloader
    # NOTE: you may have additional outputs from the loader __getitem__, you can modify this
    for (inputs, outputs) in loader:
        action_preds = []
        target_preds = []
        preds = []
        action_labels = []
        target_labels = []
        labels = []
        print("inputs and labels shape", inputs.size(), outputs.size())
        # put model inputs to device
        inputs, outputs = inputs.to(device).long(), outputs.to(device).long()
        # calculate the loss and train accuracy and perform backprop
        # NOTE: feel free to change the parameters to the model forward pass here + outputs
        loss = 0
        use_teacher_forcing = False
        if training:
            use_teacher_forcing = True if random.random() < 0.5 else False
        print("batch size:", len(inputs))
        batch_size = len(inputs) #change this
        hidden = torch.zeros(1,args.embedding_dim, requires_grad=True)
        encoder_outputs = torch.zeros(batch_size, args.instruction_length, args.embedding_dim, device=device)
        for i in range(args.instruction_length):
            # print("encoder input", inputs[:,i].size())
            encoder_output, encoder_hidden = model.encoder(inputs[:,i], hidden)
            encoder_outputs[:,i] = encoder_output
        decoder_hidden = torch.zeros(1, 684, 1)
        decoder_input = torch.zeros(batch_size, 2).long()
        # final_outputs = torch.zeros(args.targets_length, 2)
        if use_teacher_forcing:
            for i in range(args.targets_length):
                decoder_output, decoder_hidden = model.decoder(
                decoder_input, decoder_hidden, encoder_outputs)
                actions = decoder_output[:,:args.num_actions].float()
                targets = decoder_output[:,args.num_actions:].float()
                loss += criterion(actions, outputs[:,i,0]) + criterion(targets, outputs[:,i,1])
                decoder_input = outputs[:,i]  # Teacher forcing
                if decoder_input == torch.Tensor( [model.actions_to_index["Stop"], model.targets_to_index["Stop"]]):
                    break
        else:
            for i in range(args.targets_length):
                decoder_output, decoder_hidden = model.decoder(
                decoder_input, decoder_hidden, encoder_outputs)
                actions = decoder_output[:,:args.num_actions]
                targets = decoder_output[:,args.num_actions:]
                v, a = actions.topk(1)
                v, t = targets.topk(1)
                # final_outputs[i] = torch.Tensor([a, t])
                decoder_input = torch.Tensor(torch.concat((a, t), dim=1)).detach()
                loss += criterion(actions, outputs[:,i,0]) + criterion(targets, outputs[:,i,1])
                action_preds.extend(a.cpu().numpy())
                target_preds.extend(t.cpu().numpy())
                action_labels.extend(outputs[:,i,0].cpu().numpy())
                target_labels.extend(outputs[:,i,1].cpu().numpy())
                preds.append([a, t])
                labels.append([outputs[:,i,0], outputs[:,i,1]])
                if a == model.actions_to_index["Stop"] and t == model.targets_to_index["Stop"]:
                    break
        action_acc = accuracy_score(action_preds, action_labels)
        target_acc = accuracy_score(target_preds, target_labels)

        # step optimizer and compute gradients during training
        if training:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        """
        # TODO: implement code to compute some other metrics between your predicted sequence
        # of (action, target) labels vs the ground truth sequence. We already provide 
        # exact match and prefix exact match. You can also try to compute longest common subsequence.
        # Feel free to change the input to these functions.
        """
        # TODO: add code to log these metrics

        acc = (action_acc + target_acc) / 2.0
        prefix_em = (preds, labels)
        # logging
        epoch_loss += loss
        epoch_acc += acc.item()

    epoch_loss /= len(loader)
    epoch_acc /= len(loader)

    return epoch_loss, epoch_acc


def validate(args, model, loader, optimizer, criterion, device):
    # set model to eval mode
    model.eval()

    # don't compute gradients
    with torch.no_grad():
        val_loss, val_acc = train_epoch(
            args,
            model,
            loader,
            optimizer,
            criterion,
            device,
            training=False,
        )

    return val_loss, val_acc


def train(args, model, loaders, optimizer, criterion, device):
    # Train model for a fixed number of epochs
    # In each epoch we compute loss on each sample in our dataset and update the model
    # weights via backpropagation
    model.train()
    for epoch in tqdm.tqdm(range(args.num_epochs)):

        # train single epoch
        # returns loss for action and target prediction and accuracy
        train_loss, train_acc = train_epoch(
            args,
            model,
            loaders["train"],
            optimizer,
            criterion,
            device,
            training=True,
        )

        # some logging
        print(f"train loss : {train_loss}")

        # run validation every so often
        # during eval, we run a forward pass through the model and compute
        # loss and accuracy but we don't update the model weights
        if epoch % args.val_every == 0:
            val_loss, val_acc = validate(
                args,
                model,
                loaders["val"],
                optimizer,
                criterion,
                device,
            )

            print(f"val loss : {val_loss} | val acc: {val_acc}")

    # ================== TODO: CODE HERE ================== #
    # Task: Implement some code to keep track of the model training and
    # evaluation loss. Use the matplotlib library to plot
    # 3 figures for 1) training loss, 2) validation loss, 3) validation accuracy
    # ===================================================== #


def main(args):
    device = get_device(args.force_cpu)

    # get dataloaders
    train_loader, val_loader, maps, act_maps = setup_dataloader(args)
    loaders = {"train": train_loader, "val": val_loader}

    # build model
    model = setup_model(args, maps, act_maps)

    # get optimizer and loss functions
    criterion, optimizer = setup_optimizer(args, model)
    
    if args.eval:
        val_loss, val_acc = validate(
            args,
            model,
            loaders["val"],
            optimizer,
            criterion,
            device,
        )
    else:
        training = True
        train(args, model, loaders, optimizer, criterion, device)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--in_data_fn", type=str, help="data file")
    parser.add_argument(
        "--model_output_dir", type=str, help="where to save model outputs"
    )
    parser.add_argument(
        "--batch_size", type=int, default=256, help="size of each batch in loader"
    )
    parser.add_argument("--force_cpu", action="store_true", help="debug mode")
    parser.add_argument("--eval", action="store_true", help="run eval")
    parser.add_argument("--num_epochs", type=int, default=1000, help="number of training epochs")
    parser.add_argument(
        "--val_every", default=5, type=int, help="number of epochs between every eval loop"
    )
    parser.add_argument("--input_size", type=int, default=1000)
    parser.add_argument("--output_size", type=int, default=92)
    parser.add_argument("--vocab_size", type=int, default=1000)
    parser.add_argument("--embedding_dim", type=int, default=31)
    parser.add_argument("--learning_rate", type=float, default=0.01)
    parser.add_argument("--num_outputs", type=int, default=88)
    parser.add_argument("--teacher_ratio", type=float, default=0.5)
    parser.add_argument("--episode_length", type=int, default=19)
    parser.add_argument("--instruction_length", type=int, default=18962)
    parser.add_argument("--targets_length", type=int, default=20)
    parser.add_argument("--num_actions", type=int, default=9)
    
    # ================== TODO: CODE HERE ================== #
    # Task (optional): Add any additional command line
    # parameters you may need here
    # ===================================================== #
    args = parser.parse_args()

    main(args)
