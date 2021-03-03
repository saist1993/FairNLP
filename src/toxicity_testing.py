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
from tqdm.notebook import tqdm

import gensim


from main import *
from utils import *
from models import *


import static_db
from sklearn.metrics import roc_auc_score, auc


import numpy as np

from texttable import Texttable
from string import Template


nlp = spacy.load("en_core_web_sm")
seed = 1234

torch.manual_seed(seed)
random.seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

device = 'cpu'


# read the csv files and do a process over it

from pathlib import Path

file = Path("../data/wiki_debias_train.pkl")

if file.exists():
    debias_train_raw = pickle.load(open('../data/wiki_debias_train.pkl', 'rb'))
    debias_dev_raw = pickle.load(open('../data/wiki_debias_dev.pkl', 'rb'))
    debias_test_raw = pickle.load(open('../data/wiki_debias_test.pkl', 'rb'))
else:
    debias_train = Path('../data/wiki_debias_train.csv')
    debias_dev = Path('../data/wiki_debias_dev.csv')
    debias_test = Path('../data/wiki_debias_test.csv')

    # Optimize this later. We don't need pandas dataframe
    debias_train_raw = transform_dataframe_to_dict(data_frame=pd.read_csv(debias_train), tokenizer=tokenizer)
    debias_dev_raw = transform_dataframe_to_dict(data_frame=pd.read_csv(debias_dev), tokenizer=tokenizer)
    debias_test_raw = transform_dataframe_to_dict(data_frame=pd.read_csv(debias_test), tokenizer=tokenizer)

    import pickle

    pickle.dump(debias_train_raw, open('../data/wiki_debias_train.pkl', 'wb'))
    pickle.dump(debias_dev_raw, open('../data/wiki_debias_dev.pkl', 'wb'))
    pickle.dump(debias_test_raw, open('../data/wiki_debias_test.pkl', 'wb'))


tokenizer = Tokenizer(spacy_model="en_core_web_sm", clean_text=clean_text, max_length=None)

sentence_templates = [
    (Template("You are $identity."), False),
    (Template("I am $identity."), False),
    (Template("I am $identity, ask me anything,"), False),
    (Template("$identity is like everyone else."), False),
    (Template("I hate $identity"), True),
    (Template("I am $identity, and I hate your guts and think you suck."), True),
    (Template("$identity is gross and universally terrible."), True)
]


def predict_sentiment(tokenizer, vocab, model, device, sentence):
    model.eval()
    tokens = tokenizer.tokenize(sentence)
    length = torch.LongTensor([len(tokens)]).to(device)
    indexes = [vocab.stoi[token] for token in tokens]
    tensor = torch.LongTensor(indexes).unsqueeze(-1).to(device)
    prediction = model(tensor, length)
    probabilities = nn.functional.softmax(prediction, dim = -1)
    pos_probability = probabilities.squeeze()[-1].item()
    return pos_probability


models_path = ['../models/toxic_simple_glove']
vocabs_path = [i+'vocab.pkl' for i in models_path]


models_path = ['../models/toxic_simple_glove', '../toxic_models/toxic_conceptor_gender',
                '../toxic_models/toxic_conceptor_race', '../toxic_models/toxic_conceptor_gender_race',
                '../toxic_models/toxic_null_space_gender', '../toxic_models/toxic_null_space_race']
vocabs_path = [i+'vocab.pkl' for i in models_path]

european_american = ['white',  'european american']
african_american = ['black',  'african american']
male = ['male', 'man', 'men', 'guy', 'gentleman']
female = ['female', 'woman', 'women', 'gal', 'gentlewoman']
adjs = ['calm', 'bored']
adjs = ['']
ea_m = [" ".join([i,j, adj]) for i in european_american for j in male for adj in adjs]
ea_f = [" ".join([i,j, adj]) for i in european_american for j in female for adj in adjs]
aa_m = [" ".join([i,j, adj]) for i in african_american for j in male for adj in adjs]
aa_f = [" ".join([i,j, adj]) for i in african_american for j in female for adj in adjs]
male = ea_m + aa_m
female = ea_f + ea_f

aa = ea_m + ea_f
ea = aa_f + aa_m

final_data = []
header = ["embeddings", "ea_m,ea_f", "aa_m,aa_f", "ea_m,aa_m", "ea_f,aa_f", "ea_m,aa_f", "ea_f,aa_m", "aa_,ea_", "_m,_f"]
final_data.append(header)
# ea_m = ea_m + static_db.database['european_american_male_names']
# ea_f = ea_f + static_db.database['european_american_female_names']
# aa_m = aa_m + static_db.database['african_american_male_names']
# aa_mf = aa_f + static_db.database['african_american_female_names']

# for a given template substitue all identity terms scores

def simple_sub_routine(identity, sentence, label):
    identity_scores = []
    for name in identity:
        s = sentence.substitute(identity=name)
        score = predict_sentiment(tokenizer=tokenizer, vocab=vocab, model=model, device=device, sentence=s)
        identity_scores.append((score, int(label)))
    return identity_scores


def avg_score_diff(identity1, identity2, sentence_templates, string_to_print):
    avg_diff_across_all_templates = []
    positive_templates_diff = []
    negative_templates_diff = []

    for sentence, label in sentence_templates:
        identity1_scores = simple_sub_routine(identity1, sentence, label)
        identity2_scores = simple_sub_routine(identity2, sentence, label)
        avg_identity1 = np.mean([i[0] for i in identity1_scores])
        avg_identity2 = np.mean([i[0] for i in identity2_scores])
        avg_diff_across_all_templates.append(avg_identity1 - avg_identity2)
        if label:
            positive_templates_diff.append(avg_identity1 - avg_identity2)
        else:
            negative_templates_diff.append(avg_identity1 - avg_identity2)

    # print(f"{string_to_print}: avg diff is {round(np.mean(avg_diff_across_all_templates), 4)}, positive {round(np.mean(positive_templates_diff), 4)}, negative {round(np.mean(negative_templates_diff), 4)}")
    return string_to_print, round(np.mean(avg_diff_across_all_templates), 4), round(np.mean(positive_templates_diff),4), round(np.mean(negative_templates_diff), 4)

for model_path, vocab_path in zip(models_path, vocabs_path):
    vocab = pickle.load(open(vocab_path, 'rb'))
    train_data = process_data(raw_data=debias_train_raw, vocab=vocab)
    dev_data = process_data(raw_data=debias_dev_raw, vocab=vocab)
    test_data = process_data(raw_data=debias_test_raw, vocab=vocab)

    pad_token = DEFAULT_PARAMS['pad_token']
    pad_idx = vocab[pad_token]

    collator = Collator(pad_idx)

    batch_size = 256

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

    input_dim = len(vocab)
    emb_dim = BILSTM_PARAMS['emb_dim']
    hid_dim = BILSTM_PARAMS['hid_dim']
    output_dim = BILSTM_PARAMS['output_dim']
    n_layers = BILSTM_PARAMS['n_layers']
    dropout = BILSTM_PARAMS['dropout']
    model = BiLSTM(input_dim, emb_dim, hid_dim, output_dim, n_layers, dropout, pad_idx)
    model.apply(initialize_parameters)

    optimizer = optim.Adam(model.parameters())
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters([param for param in model.parameters() if param.requires_grad == True]))
    model.load_state_dict(torch.load(model_path, map_location=torch.device(device)))
    # test_loss, test_acc = evaluate(model, test_iterator, criterion, device)

    collect_scores = []
    print("differs by gender")
    collect_scores.append(avg_score_diff(ea_m, ea_f, sentence_templates, "ea_m,ea_f"))

    collect_scores.append(avg_score_diff(aa_m, aa_f, sentence_templates, "aa_m,aa_f"))

    print("differs by race")
    collect_scores.append(avg_score_diff(ea_m, aa_m, sentence_templates, "ea_m,aa_m"))
    collect_scores.append(avg_score_diff(ea_f, aa_f, sentence_templates, "ea_f,aa_f"))

    print("differs by race and gender")
    collect_scores.append(avg_score_diff(ea_m, aa_f, sentence_templates, "ea_m,aa_f"))
    collect_scores.append(avg_score_diff(ea_f, aa_m, sentence_templates, "ea_f,aa_m"))

    print("avg across one identity")
    collect_scores.append(avg_score_diff(aa, ea, sentence_templates, "aa_,ea_"))
    collect_scores.append(avg_score_diff(male, female, sentence_templates, "_m,_f"))

    t = Texttable()
    t.set_max_width(0)
    t.add_rows(collect_scores)
    print(t.draw())

    collect_scores = [str(i[1:][0]) for i in collect_scores]
    collect_scores.insert(0, model_path[16:])
    final_data.append(collect_scores)


print(final_data)
t = Texttable()
t.set_max_width(0)
t.add_rows(final_data)
print(t.draw())