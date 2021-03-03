# A file to play around with the task 1 of SemEval 2018
from string import Template
from pathlib import Path

import gensim
import time



import torch
import torch.nn as nn

import torch.optim as optim
import scipy


from tqdm import tqdm
from torchtext.experimental.datasets.text_classification import TextClassificationDataset
from torchtext.experimental.functional import sequential_transforms, vocab_func, totensor

from models import *
import static_db
from main import clean_text, Tokenizer, build_vocab_from_data, get_pretrained_embedding, epoch_time
from utils import parse_args

import random
import sys

import pickle

from nltk.tokenize import TweetTokenizer


def custom_clean_text(text):
    return text.replace('#', '').replace('@', '')

class CustomTweetTokenizer:
    """Cleans the data and tokenizes it"""

    def __init__(self, spacy_model: str = "en_core_web_sm", clean_text=clean_text, max_length=None):
        self.tokenizer_model = TweetTokenizer()
        self.clean_text = clean_text
        self.max_length = max_length

    def tokenize(self, s):

        tokens = self.tokenizer_model.tokenize(s)
        final_token = []
        if self.clean_text:
            for t in tokens:
                clean_t = clean_text(t)
                if clean_t:
                    final_token.append(clean_t)

            tokens = final_token
        # tokens = [token.text for token in doc]

        if self.max_length:
            tokens = tokens[:self.max_length]

        return tokens

    def batch_tokenize(self, texts: list):
        """tokenizes a list via nlp pipeline space"""
        # nlp = self.tokenizer_model

        tokenized_list = []
        for t in texts:
            tokenized_list.append(self.tokenize(t))
        return tokenized_list
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
    "emb_dim": 200,
    "hid_dim": 256,
    "output_dim": 2,
    "n_layers": 2,
    "dropout": 0.5
}

params = DEFAULT_PARAMS
bilstm_params = BILSTM_PARAMS

data_dir = Path('../data/affect_in_tweet/task1/V-reg/')
emotions = ['anger', 'fear', 'joy', 'sadness']
splits = ['train', 'dev', 'test-gold']
prepend_text = Template('Valence-reg-En-$split.txt')


def read_data(data_dir, split):
    a = open(data_dir/Path(prepend_text.substitute(split=split)), 'r')
    lines = a.readlines()
    final_data = [l.strip().split('\t') for l in lines[1:]]
    return final_data

def remove_data_without_label(data):
    new_data = []
    for d in data:
        if len(d[1]) > 5 and  d[-1] != 'NONE' and float(d[-1]) > 0.001:
            new_data.append(d)
    return new_data


def transform_dataframe_to_dict(data, tokenizer):
    final_data = []
    for i in data:
        if len(clean_text(i[1])) >= 5:
            temp = {
                'original_text': i[1],
                'emotion': i[2],
                'toxicity': float(i[-1]),
                'text': clean_text(i[1])
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

def process_data(raw_data, vocab):
    """
    raw data is assumed to be tokenized
    Can't use the main one as this needs to be manipulated
    """
    final_data = [(data_point['toxicity'], data_point['tokenized_text']) for data_point in raw_data]
    text_transformation = sequential_transforms(vocab_func(vocab),
                                                totensor(dtype=torch.long))
    label_transform = sequential_transforms(totensor(dtype=torch.float))

    transforms = (label_transform, text_transformation)

    return TextClassificationDataset(final_data, vocab, transforms)

class Collator:
    def __init__(self, pad_idx):
        self.pad_idx = pad_idx

    def collate(self, batch):
        labels, text = zip(*batch)
        labels = torch.FloatTensor(labels)
        lengths = torch.LongTensor([len(x) for x in text])
        text = nn.utils.rnn.pad_sequence(text, padding_value=self.pad_idx)

        return labels, text, lengths

def calculate_accuracy(predictions, labels):
    return scipy.stats.pearsonr(predictions.squeeze().detach().cpu().numpy(), labels.cpu().detach().cpu().numpy())[0]

def train_loop(model, iterator, optimizer, criterion, device):
    epoch_loss = 0
    epoch_acc = 0

    model.train()

    for labels, text, lengths in tqdm(iterator):
        labels = labels.to(device)
        text = text.to(device)

        optimizer.zero_grad()

        predictions = model(text, lengths)

        loss = criterion(predictions.squeeze(), labels.squeeze())

        acc = calculate_accuracy(predictions, labels)

        loss.backward()

        optimizer.step()

        epoch_loss += loss.item()
        epoch_acc += acc.item()

    return epoch_loss / len(iterator), epoch_acc / len(iterator)


def evaluate_loop(model, iterator, criterion, device):
    epoch_loss = 0
    epoch_acc = 0

    model.eval()

    with torch.no_grad():
        for labels, text, lengths in iterator:
            labels = labels.to(device)
            text = text.to(device)

            predictions = model(text, lengths)

            loss = criterion(predictions.squeeze(), labels.squeeze())

            acc = calculate_accuracy(predictions, labels)

            epoch_loss += loss.item()
            epoch_acc += acc.item()

    return epoch_loss / len(iterator), epoch_acc / len(iterator)



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

    train = read_data(data_dir, 'train')

    dev = read_data(data_dir, 'dev')

    test = read_data(data_dir, 'test-gold')

    print(f"length of train is {len(train)}, length of dev is {len(dev)}, and length of test is {len(test)}")

    train = remove_data_without_label(train)
    dev = remove_data_without_label(dev)
    test = remove_data_without_label(test)

    print(
        f"After cleanup: length of train is {len(train)}, length of dev is {len(dev)}, and length of test is {len(test)}")

    tokenizer = CustomTweetTokenizer(spacy_model="en_core_web_sm", clean_text=custom_clean_text, max_length=None)

    dev_processed = transform_dataframe_to_dict(data=dev, tokenizer=tokenizer)
    train_processed = transform_dataframe_to_dict(data=train, tokenizer=tokenizer)
    test_processed = transform_dataframe_to_dict(data=test, tokenizer=tokenizer)

    temp_vocab = build_vocab_from_data(raw_train_data=train_processed, raw_dev_data=dev_processed)


    groups = ['male_names', 'female_names',
              'african_american_names', 'european_american_names',
              'african_american_male_names', 'african_american_female_names',
              'european_american_male_names', 'european_american_female_names']
    combined_names = list(set([j for i in groups for j in static_db.database[i]]))
    names_to_add = []
    for c in combined_names:
        if temp_vocab.stoi[c] == 0:
            names_to_add.append(c)

    vocab = build_vocab_from_data(raw_train_data=train_processed, raw_dev_data=dev_processed,
                                  artificial_populate=names_to_add)

    # prepare training data and define transformation function
    train_data = process_data(raw_data=train_processed, vocab=vocab)
    dev_data = process_data(raw_data=dev_processed, vocab=vocab)
    test_data = process_data(raw_data=test_processed, vocab=vocab)

    pad_token = params['pad_token']
    pad_idx = vocab[pad_token]
    collator = Collator(pad_idx)

    batch_size = params['batch_size']
    pickle.dump(vocab, open(params['model_save_name'] + 'vocab.pkl', 'wb'))
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
        output_dim = 1
        n_layers = bilstm_params['n_layers']
        dropout = bilstm_params['dropout']

        model = BiLSTM(input_dim, emb_dim, hid_dim, output_dim, n_layers, dropout, pad_idx)
        model.apply(initialize_parameters)


        criterion = nn.MSELoss()

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

    model.embedding.weight.requires_grad = False

    optimizer = optim.Adam(model.parameters([param for param in model.parameters() if param.requires_grad == True]))

    n_epochs = 200

    best_valid_loss = float('inf')

    for epoch in range(n_epochs):

        start_time = time.monotonic()

        train_loss, train_acc = train_loop(model, train_iterator, optimizer, criterion, device)
        valid_loss, valid_acc = evaluate_loop(model, dev_iterator, criterion, device)
        test_loss, test_acc = evaluate_loop(model, test_iterator, criterion, device)

        end_time = time.monotonic()

        epoch_mins, epoch_secs = epoch_time(start_time, end_time)

        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), params['model_save_name'])

        print(f'Epoch: {epoch + 1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s')
        print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc:.2f}%')
        print(f'\t Val. Loss: {valid_loss:.3f} |  Val. Acc: {valid_acc:.2f}%')
        print(f'\t Test Loss: {test_loss:.3f} |  Test Acc: {test_acc:.2f}%')