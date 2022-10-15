import argparse
import os
import tqdm
import torch
# from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader, Dataset
import random
from eval_utils import downstream_validation
import utils
import data_utils
import numpy as np

from skipmodel import skip

verbose = False
class skip_data(Dataset):
    def __init__(self, args, dataset):
        self.dataset = dataset
        self.len = len(dataset)

    def __getitem__(self, idx):
        input, context = self.dataset[idx]
        return torch.tensor(input, dtype=torch.float32), torch.tensor(context)

    def __len__(self):
        return self.len

def setup_dataloader(args):
    """
    return:
        - train_loader: torch.utils.data.Dataloader
        - val_loader: torch.utils.data.Dataloader
    """

    # read in training data from books dataset
    sentences = data_utils.process_book_dir(args.data_dir)
    # build one hot maps for input and output
    (
        vocab_to_index,
        index_to_vocab,
        suggested_padding_len,
    ) = data_utils.build_tokenizer_table(sentences, vocab_size=args.vocab_size)

    # create encoded input and output numpy matrices for the entire dataset and then put them into tensors
    encoded_sentences, lens = data_utils.encode_data(
        sentences,
        vocab_to_index,
        suggested_padding_len,
    )

    # ================== TODO: CODE HERE ================== #
    # Task: Given the tokenized and encoded text, you need to
    # create inputs to the LM model you want to train.
    # E.g., could be target word in -> context out or
    # context in -> target word out.
    # You can build up that input/output table across all
    # encoded sentences in the dataset!
    # Then, split the data into train set and validation set
    # (you can use utils functions) and create respective
    # dataloaders.
    # ===================================================== #
    dataset_t = []
    dataset_v = []
    didx = 0
    for idx, sentence in enumerate(encoded_sentences):
        length = lens[idx][0]
        for idx, word in enumerate(sentence):      
            begin = idx - args.context_window
            end = idx + args.context_window + 1
            context_words = []
            for i in range(begin, end):
                if i != idx:
                    if 0 <= i < length:
                        context_words.append(sentence[i])
                    else:
                        context_words.append(0)
            # [sentence[i] for i in range(begin, end) if 0 <= i < length and i != idx ]
            if sum(context_words) == 0:
                # index out of bounds, past the sentence length, or all padding
                #this can't happen in real sentences bc we throw out words of len 0
                continue
            target = word
            # [0,9] - 0.8 for train and 0.2 for test
            if torch.randint(10, (1,)) <=7:
                dataset_v.append(tuple((target, context_words)))
            else:
                dataset_t.append(tuple((target, context_words)))
            didx += 1
    length = didx
    print("len", length)
    train_set = skip_data(args, dataset_t)
    val_set = skip_data(args, dataset_v)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, drop_last=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, drop_last=True)
    return train_loader, val_loader, index_to_vocab

def setup_model(args):
    """
    return:
        - model: YourOwnModelClass
    """
    # ================== TODO: CODE HERE ================== #
    # Task: Initialize your CBOW or Skip-Gram model.
    # ===================================================== #
    model = skip(args)
    return model


def setup_optimizer(args, model):
    """
    return:
        - criterion: loss_fn
        - optimizer: torch.optim
    """
    # ================== TODO: CODE HERE ================== #
    # Task: Initialize the loss function for predictions. 
    # Also initialize your optimizer.
    # ===================================================== #
    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters())
    return criterion, optimizer

def get_accuracy(outputs: torch.Tensor, labels: torch.Tensor):
    length = len(outputs)
    loss = 0
    for i in range(length):
        loss += iou_pytorch(outputs[i].type(torch.int32), labels[i].type(torch.int32))
    return loss / length

def iou_pytorch(outputs: torch.Tensor, labels: torch.Tensor):
    SMOOTH = 1e-6
    # You can comment out this line if you are passing tensors of equal shape
    # But if you are passing output from UNet or something it will most probably
    # be with the BATCH x 1 x H x W shape
    # outputs = outputs.squeeze(1)  # BATCH x 1 x H x W => BATCH x H x W
    
    intersection = (outputs & labels).float().sum()  # Will be zero if Truth=0 or Prediction=0
    union = (outputs | labels).float().sum()         # Will be zero if both are 0
    
    iou = (intersection + SMOOTH) / (union + SMOOTH)  # We smooth our devision to avoid 0/0
    print("union, ", union)
    print("iou", iou)
    # thresholded = torch.clamp(20 * (iou - 0.5), 0, 10).ceil() / 10  # This is equal to comparing with thresolds
    
    return iou
    

def train_epoch(
    args,
    model,
    loader,
    optimizer,
    criterion,
    device,
    training=True,
):
    model.train()
    epoch_loss = 0.0

    # keep track of the model predictions for computing accuracy
    pred_labels = []
    target_labels = []
    idx = 0
    # iterate over each batch in the dataloader
    # NOTE: you may have additional outputs from the loader __getitem__, you can modify this
    for (inputs, labels) in tqdm.tqdm(loader):
        if verbose:
            print("inputs, labels", inputs.size(), labels.size())
        if verbose:
            print("labels", labels[0])
        # put model inputs to device
        labels = torch.nn.functional.one_hot(labels.to(torch.int64), 3000) #number of classes is 3000
        if verbose:
            print("labels", labels)
        labels = torch.sum(labels, dim=1)
        if verbose:
            print("labels", labels.size())
        inputs.requires_grad = True
        inputs, labels = inputs.to(device).long(), labels.to(device).float()

        # calculate the loss and train accuracy and perform backprop
        # NOTE: feel free to change the parameters to the model forward pass here + outputs
        pred_logits = model(inputs)
        if verbose:
            print("pred_logits", pred_logits.size())
        values, indices = torch.topk(pred_logits, args.context_window*2)
        hot_logits = torch.nn.functional.one_hot(indices, 3000)
        hot_logits = torch.sum(hot_logits, dim=1).type(dtype=torch.float32)
        if verbose:
            print("hot_logits", hot_logits.size())
        # calculate prediction loss
        if verbose:
            print("hot_logits, labels", hot_logits.type(),labels.type() )
        hot_logits.requires_grad = True
        loss = criterion(hot_logits, labels)

        # step optimizer and compute gradients during training
        if training:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # logging
        epoch_loss += loss.item()

        # compute metrics
        pred_labels.append(hot_logits)
        target_labels.append(labels)

    acc = get_accuracy(pred_labels, target_labels)
    epoch_loss /= len(loader)

    return epoch_loss, acc


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

def main(args):
    options = vars(args)
    device = utils.get_device(args.force_cpu)

    # load analogies for downstream eval
    external_val_analogies = utils.read_analogies(args.analogies_fn)

    if args.downstream_eval:
        word_vec_file = os.path.join(args.output_dir, args.word_vector_fn)
        assert os.path.exists(word_vec_file), "need to train the word vecs first!"
        downstream_validation(word_vec_file, external_val_analogies)
        return

    # get dataloaders
    train_loader, val_loader, index_to_vocab = setup_dataloader(args)

    loaders = {"train": train_loader, "val": val_loader}

    # build model
    model = setup_model(args)

    # get optimizer
    criterion, optimizer = setup_optimizer(args, model)

    for epoch in range(args.num_epochs):
        # train model for a single epoch
        print(f"Epoch {epoch}")
        train_loss, train_acc = train_epoch(
            args,
            model,
            loaders["train"],
            optimizer,
            criterion,
            device,
        )

        print(f"train loss : {train_loss} | train acc: {train_acc}")

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

            # ======================= NOTE ======================== #
            # Saving the word vectors to disk and running the eval
            # can be costly when you do it multiple times. You could
            # change this to run only when your training has concluded.
            # However, incremental saving means if something crashes
            # later or you get bored and kill the process you'll still
            # have a word vector file and some results.
            # ===================================================== #

            # save word vectors
            word_vec_file = os.path.join(args.output_dir, args.word_vector_fn)
            print("saving word vec to ", word_vec_file)
            utils.save_word2vec_format(word_vec_file, model, index_to_vocab)

            # evaluate learned embeddings on a downstream task
            downstream_validation(word_vec_file, external_val_analogies)

        if epoch % args.save_every == 0:
            ckpt_file = os.path.join(args.output_dir, "model.ckpt")
            print("saving model to ", ckpt_file)
            torch.save(model, ckpt_file)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, help="where to save training outputs", default='./skipsaved')
    parser.add_argument("--data_dir", type=str, help="where the book dataset is stored")
    parser.add_argument(
        "--downstream_eval",
        action="store_true",
        help="run downstream eval on trained word vecs",
    )
    # ======================= NOTE ======================== #
    # If you adjust the vocab_size down below 3000, there 
    # may be analogies in the downstream evaluation that have
    # words that are not in your vocabulary, resulting in
    # automatic (currently) zero score for an ABCD where one
    # of A, B, C, or D is not in the vocab. A visible warning
    # will be generated by the evaluation loop for these examples.
    # ===================================================== #
    parser.add_argument(
        "--vocab_size", type=int, default=3000, help="size of vocabulary"
    )
    parser.add_argument(
        "--batch_size", type=int, default=256, help="size of each batch in loader"
    )
    parser.add_argument("--force_cpu", action="store_true", help="debug mode")
    parser.add_argument(
        "--analogies_fn", type=str, help="filepath to the analogies json file"
    )
    parser.add_argument(
        "--word_vector_fn", type=str, help="filepath to store the learned word vectors",
        default='learned_word_vectors.txt'
    )
    parser.add_argument(
        "--num_epochs", default=30, type=int, help="number of training epochs"
    )
    parser.add_argument(
        "--val_every",
        default=5,
        type=int,
        help="number of epochs between every eval loop",
    )
    parser.add_argument(
        "--save_every",
        default=5,
        type=int,
        help="number of epochs between saving model checkpoint",
    )
    parser.add_argument(
        "--context_window",
        default=1,
        type=int,
        help="how many words the cbow sees before and after the target word",
    )
    # ================== TODO: CODE HERE ================== #
    # Task (optional): Add any additional command line
    # parameters you may need here
    # ===================================================== #

    args = parser.parse_args()
    main(args)