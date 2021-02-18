import torch
import torch.nn as nn

import torch.optim as optim

import torchtext
import torchtext.experimental
import torchtext.experimental.vectors
from torchtext.experimental.datasets.raw.text_classification import RawTextIterableDataset
from torchtext.experimental.datasets.text_classification import TextClassificationDataset
from torchtext.experimental.functional import sequential_transforms, vocab_func, totensor

import collections
import random
import time

import pandas as pd

from pathlib import Path

import spacy
import re

from functools import partial
from tqdm import tqdm
import sys

import gensim




from models import BiLSTM, initialize_parameters


from utils  import parse_args, get_pretrained_embedding

DEFAULT_PARAMS = {
    "spacy_model": "en_core_web_sm",
    "seed": 1234,
    "train_path": "../data/wiki_debias_train.csv",
    "dev_path": "../data/wiki_debias_dev.csv",
    "test_path": "../data/wiki_debias_test.csv",
    "batch_size" : 512,
    "pad_token": '<pad>',
    "unk_token": "<unk>",
    "device": "cpu",
    "pre_trained_word_vec": "../data/testvec1",
    "model_save_name": "bilstm.pt",
    "model": "bilstm",
}

BILSTM_PARAMS = {
    "emb_dim": 300,
    "hid_dim": 256,
    "output_dim": 2,
    "n_layers": 2,
    "dropout": 0.5
}


def transform_dataframe_to_dict(data_frame, tokenizer):
    final_data = []
    for i in data_frame.iterrows():
        if len(clean_text(i[1]['comment'])) >= 5:
            temp = {
                'original_text': i[1]['comment'],
                'lable': i[1]['is_toxic'],
                'toxicity': i[1]['toxicity'],
                'text': clean_text(i[1]['comment'])
            }
            final_data.append(temp)
    texts = [f['text'] for f in final_data]
    tokenized_text = tokenizer.batch_tokenize(texts=texts)
    new_final_data = []
    for index, f in enumerate(final_data):
        f['tokenized_text'] = tokenized_text[index]
        if len(tokenized_text[index]) > 1:
            new_final_data.append(f)

    # sanity check
    for f in new_final_data:
        assert len(f['tokenized_text']) > 1

    return new_final_data


def clean_text(text: str):
    """
    cleans text casing puntations and special characters. Removes extra space
    """
    text = re.sub('[^ a-zA-Z0-9]|unk', '', text)
    text = text.strip()
    return text


class Tokenizer:
    """Cleans the data and tokenizes it"""

    def __init__(self, spacy_model: str = "en_core_web_sm", clean_text=clean_text, max_length=None):
        self.tokenizer_model = spacy.load(spacy_model)
        self.clean_text = clean_text
        self.max_length = max_length

    def tokenize(self, s):
        if self.clean_text:
            s = clean_text(s)
        doc = self.tokenizer_model(s)
        tokens = [token.text for token in doc]

        if self.max_length:
            tokens = tokens[:self.max_length]

        return tokens

    def batch_tokenize(self, texts: list, ):
        """tokenizes a list via nlp pipeline space"""
        nlp = self.tokenizer_model

        tokenized_list = []

        if self.max_length:
            for doc in tqdm(nlp.pipe(texts, disable=["ner", "tok2vec"])):
                tokenized_list.append([t.text for t in doc][:self.max_length])
        else:
            for doc in tqdm(nlp.pipe(texts, disable=["ner", "tok2vec"])):
                tokenized_list.append([t.text for t in doc])

        return tokenized_list


def build_vocab_from_data(raw_train_data, raw_dev_data):
    """This has been made customly for the given dataset. Need to write your own for any other use case"""

    token_freqs = collections.Counter()
    for data_point in raw_train_data:
        token_freqs.update(data_point['tokenized_text'])
    for data_point in raw_dev_data:
        token_freqs.update(data_point['tokenized_text'])
    #     token_freqs.update(data_point['tokenized_text'] for data_point in raw_train_data)
    #     token_freqs.update(data_point['tokenized_text'] for data_point in raw_dev_data)
    vocab = torchtext.vocab.Vocab(token_freqs)
    return vocab


def process_data(raw_data, vocab):
    """raw data is assumed to be tokenized"""
    final_data = [(data_point['lable'], data_point['tokenized_text']) for data_point in raw_data]
    text_transformation = sequential_transforms(vocab_func(vocab),
                                                totensor(dtype=torch.long))
    label_transform = sequential_transforms(totensor(dtype=torch.long))

    transforms = (label_transform, text_transformation)

    return TextClassificationDataset(final_data, vocab, transforms)


class Collator:
    def __init__(self, pad_idx):
        self.pad_idx = pad_idx

    def collate(self, batch):
        labels, text = zip(*batch)
        labels = torch.LongTensor(labels)
        lengths = torch.LongTensor([len(x) for x in text])
        text = nn.utils.rnn.pad_sequence(text, padding_value=self.pad_idx)

        return labels, text, lengths

def calculate_accuracy(predictions, labels):
    top_predictions = predictions.argmax(1, keepdim = True)
    correct = top_predictions.eq(labels.view_as(top_predictions)).sum()
    accuracy = correct.float() / labels.shape[0]
    return accuracy


def train(model, iterator, optimizer, criterion, device):
    epoch_loss = 0
    epoch_acc = 0

    model.train()

    for labels, text, lengths in tqdm(iterator):
        labels = labels.to(device)
        text = text.to(device)

        optimizer.zero_grad()

        predictions = model(text, lengths)

        loss = criterion(predictions, labels)

        acc = calculate_accuracy(predictions, labels)

        loss.backward()

        optimizer.step()

        epoch_loss += loss.item()
        epoch_acc += acc.item()

    return epoch_loss / len(iterator), epoch_acc / len(iterator)


def evaluate(model, iterator, criterion, device):
    epoch_loss = 0
    epoch_acc = 0

    model.eval()

    with torch.no_grad():
        for labels, text, lengths in iterator:
            labels = labels.to(device)
            text = text.to(device)

            predictions = model(text, lengths)

            loss = criterion(predictions, labels)

            acc = calculate_accuracy(predictions, labels)

            epoch_loss += loss.item()
            epoch_acc += acc.item()

    return epoch_loss / len(iterator), epoch_acc / len(iterator)

def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


if __name__ == '__main__':
    parsed_args = parse_args(sys.argv[1:])
    params = DEFAULT_PARAMS.copy()
    bilstm_params = BILSTM_PARAMS.copy()

    for k,v in parsed_args.items():
        if k.lower() in params.keys():
            params[k.lower()] = v
        elif k.lower() in bilstm_params:
            bilstm_params[k.lower()] = v

    # defaults to LSTM too
    if params['model'] == 'bilstm':
        model_params = bilstm_params

    seed = params['seed']

    torch.manual_seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    print("creating tokenizer")
    tokenizer = Tokenizer(spacy_model="en_core_web_sm", clean_text=clean_text, max_length=-1)


    # read the csv files and do a process over it

    debias_train = Path('../data/wiki_debias_train.csv')
    debias_dev = Path('../data/wiki_debias_dev.csv')
    debias_test = Path('../data/wiki_debias_test.csv')

    print("reading and tokenizing data. This is going to take long time: 10 mins ")
    # Optimize this later. We don't need pandas dataframe
    debias_train_raw = transform_dataframe_to_dict(data_frame=pd.read_csv(debias_train), tokenizer=tokenizer)
    debias_dev_raw = transform_dataframe_to_dict(data_frame=pd.read_csv(debias_dev), tokenizer=tokenizer)
    debias_test_raw = transform_dataframe_to_dict(data_frame=pd.read_csv(debias_test), tokenizer=tokenizer)
    print("done tokenizing")



    # vocab object of torch text. Has inbuilt functionalities like stoi
    vocab = build_vocab_from_data(raw_train_data=debias_train_raw, raw_dev_data=debias_dev_raw)

    # prepare training data and define transformation function
    train_data = process_data(raw_data=debias_train_raw, vocab=vocab)
    dev_data = process_data(raw_data=debias_dev_raw, vocab=vocab)
    test_data = process_data(raw_data=debias_test_raw, vocab=vocab)

    pad_token = params['pad_token']
    pad_idx = vocab[pad_token]
    collator = Collator(pad_idx)

    batch_size = params['batch_size']

    print("creating training data iterators")
    train_iterator = torch.utils.data.DataLoader(train_data,
                                                 batch_size,
                                                 shuffle=True,
                                                 collate_fn=collator.collate)

    dev_iterator = torch.utils.data.DataLoader(dev_data,
                                               batch_size,
                                               shuffle=False,
                                               collate_fn=collator.collate)

    test_iterator = torch.utils.data.DataLoader(test_data,
                                                batch_size,
                                                shuffle=False,
                                                collate_fn=collator.collate)

    if params['model'] == 'bilstm':
        input_dim = len(vocab)
        emb_dim = bilstm_params['emb_dim']
        hid_dim = bilstm_params['hid_dim']
        output_dim = bilstm_params['output_dim']
        n_layers = bilstm_params['n_layers']
        dropout = bilstm_params['dropout']

        model = BiLSTM(input_dim, emb_dim, hid_dim, output_dim, n_layers, dropout, pad_idx)
        model.apply(initialize_parameters)

    optimizer = optim.Adam(model.parameters())
    criterion = nn.CrossEntropyLoss()
    device = torch.device(params['device'])



    print("reading new vector file")
    pretrained_embedding = gensim.models.KeyedVectors.load_word2vec_format(params['pre_trained_word_vec'])
    pretrained_vocab = [key for key in pretrained_embedding.vocab.keys()]



    print("updating embeddings")
    unk_token = '<unk>'
    pretrained_embedding, unk_tokens = get_pretrained_embedding(initial_embedding=model.embedding,
                                                                pretrained_vocab=pretrained_vocab,
                                                                pretrained_vectors=pretrained_embedding,
                                                                vocab=vocab,
                                                                unk_token=unk_token,
                                                                device=device)

    model = model.to(device)
    model.embedding.weight.data.copy_(pretrained_embedding)

    n_epochs = 10

    best_valid_loss = float('inf')

    for epoch in range(n_epochs):

        start_time = time.monotonic()

        train_loss, train_acc = train(model, train_iterator, optimizer, criterion, device)
        valid_loss, valid_acc = evaluate(model, dev_iterator, criterion, device)

        end_time = time.monotonic()

        epoch_mins, epoch_secs = epoch_time(start_time, end_time)

        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), 'bilstm-model.pt')

        print(f'Epoch: {epoch + 1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s')
        print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc * 100:.2f}%')
        print(f'\t Val. Loss: {valid_loss:.3f} |  Val. Acc: {valid_acc * 100:.2f}%')






