from sentiment_valence import *
# A file to play around with the task 1 of SemEval 2018
import numpy as np
import pandas as pd
from pathlib import Path
from string import Template


from main import clean_text, Tokenizer, build_vocab_from_data

import torch

import torchtext
import torch.nn as nn
import torch.optim as optim
import torchtext.experimental
import torchtext.experimental.vectors
from torchtext.experimental.datasets.raw.text_classification import RawTextIterableDataset
from torchtext.experimental.datasets.text_classification import TextClassificationDataset
from torchtext.experimental.functional import sequential_transforms, vocab_func, totensor

from models import *
from nltk.tokenize import TweetTokenizer

from texttable import Texttable


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

params = DEFAULT_PARAMS
bilstm_params = BILSTM_PARAMS
data_dir = Path('../data/affect_in_tweet/task1/V-reg/')
# emotions = ['anger', 'fear', 'joy', 'sadness']
splits = ['train', 'dev', 'test-gold']
prepend_text = Template('Valence-reg-En-$split.txt')

def read_data(data_dir, split):
    a = open(data_dir/Path(prepend_text.substitute(split=split)), 'r')
    lines = a.readlines()
    final_data = [l.strip().split('\t') for l in lines[1:]]
    return final_data



train = read_data(data_dir, 'train')

dev = read_data(data_dir, 'dev')

test = read_data(data_dir, 'test-gold')

print(f"length of train is {len(train)}, length of dev is {len(dev)}, and length of test is {len(test)}")


def remove_data_without_label(data):
    new_data = []
    for d in data:
        if len(d[1]) > 5 and  d[-1] != 'NONE' and float(d[-1]) > 0.001:
            new_data.append(d)
    return new_data

train = remove_data_without_label(train)
dev = remove_data_without_label(dev)
test = remove_data_without_label(test)

print(f"After cleanup: length of train is {len(train)}, length of dev is {len(dev)}, and length of test is {len(test)}")


seed = 1234

torch.manual_seed(seed)
random.seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

device = 'cpu'



# tokenizer = Tokenizer(spacy_model="en_core_web_sm", clean_text=False, max_length=None)
tokenizer = CustomTweetTokenizer(spacy_model="en_core_web_sm", clean_text=custom_clean_text, max_length=None)

models_path = ['../models/conceptor_gender','../models/conceptor_race', '../models/conceptor_gender_race',
              '../models/null_space_gender', '../models/null_space_race', '../models/null_space_gender_race',
              '../models/simple_glove']
vocabs_path = [i+'vocab.pkl' for i in models_path]

final_data = [['model', 'MF__', 'MF__', '__AE', '__AE', 'MMAE', 'MMAE', 'FFAE', 'FFAE', 'MFAA', 'MFAA', 'MFEE', 'MFEE', 'MFAE', 'MFAE', 'MFEA', 'MFEA']]

for model_path, vocab_path in zip(models_path, vocabs_path):
    print(f"current model under consideration {model_path}")
    # @TDOO: download the original vocab
    vocab = pickle.load(open(vocab_path, 'rb'))
    dev_processed = transform_dataframe_to_dict(data=dev, tokenizer=tokenizer)
    train_processed = transform_dataframe_to_dict(data=train, tokenizer=tokenizer)
    test_processed = transform_dataframe_to_dict(data=test, tokenizer=tokenizer)



    pad_token = params['pad_token']
    pad_idx = vocab[pad_token]
    collator = Collator(pad_idx)

    batch_size = params['batch_size']

    # prepare training data and define transformation function
    train_data = process_data(raw_data=train_processed, vocab=vocab)
    dev_data = process_data(raw_data=dev_processed, vocab=vocab)
    test_data = process_data(raw_data=test_processed, vocab=vocab)


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

    input_dim = len(vocab)
    emb_dim = 200
    hid_dim = bilstm_params['hid_dim']
    output_dim = 1
    n_layers = bilstm_params['n_layers']
    dropout = bilstm_params['dropout']

    model = BiLSTM(input_dim, emb_dim, hid_dim, output_dim, n_layers, dropout, pad_idx)
    model.apply(initialize_parameters)

    optimizer = optim.Adam(model.parameters())
    criterion = nn.MSELoss()
    device = torch.device(params['device'])

    model.load_state_dict(torch.load(model_path, map_location=torch.device(device)))

    # conceptor_gender;conceptor_gendervocab.pkl

    def predict_sentiment(tokenizer, vocab, model, device, sentence):
        model.eval()
        tokens = tokenizer.tokenize(sentence)
        length = torch.LongTensor([len(tokens)]).to(device)
        indexes = [vocab.stoi[token] for token in tokens]
        tensor = torch.LongTensor(indexes).unsqueeze(-1).to(device)
        prediction = model(tensor, length)
        #     probabilities = nn.functional.softmax(prediction, dim = -1)
        #     pos_probability = probabilities.squeeze()[-1].item()
        return prediction.detach().cpu().numpy()[0][0]


    # reading the equality corpus dataset and postprocessing it
    data_path = '../data/Equity-Evaluation-Corpus/Equity-Evaluation-Corpus/Equity-Evaluation-Corpus.csv'
    eec = pd.read_csv(data_path)

    # find templates which have both gender information and race and assigning a score to those

    new_eec = []
    for row in tqdm(eec.iterrows()):
        if type(row[1]['Race']) == str and type(row[1]['Gender']) == str:
            temp = {
                'id': row[1]['ID'],
                'sentence': row[1]['Sentence'],
                'template': row[1]['Template'],
                'person': row[1]['Person'],
                'gender': row[1]['Gender'],
                'race': row[1]['Race'],
                'emotion': row[1]['Emotion'],
                'emotion_word': row[1]['Emotion word'],
                'score': predict_sentiment(tokenizer=tokenizer, vocab=vocab, model=model, device=device,
                                           sentence=row[1]['Sentence'])
            }
            new_eec.append(temp)


    def comparision_between_two(data, group1, group2, race1, race2, print_str):
        """Assumes that group1 and group2 are gender"""
        if group1 and race1:
            group1_data = [i for i in data if i['gender'] == group1 and i['race'] == race1]
            group2_data = [i for i in data if i['gender'] == group2 and i['race'] == race2]
        elif group1:
            group1_data = [i for i in data if i['gender'] == group1]
            group2_data = [i for i in data if i['gender'] == group2]
        else:
            group1_data = [i for i in data if i['race'] == race1]
            group2_data = [i for i in data if i['race'] == race2]

        # now find corresponding pairs .. i.e they should only differ in noun phrase and nothing else
        all_unique_templates = list(set([i['template'] for i in group1_data]))
        all_unique_emotion_words = list(set([i['emotion_word'] for i in group1_data]))

        diff1 = []
        diff2 = []
        for template in all_unique_templates:
            for emotion_word in all_unique_emotion_words:
                group1_score = [i['score'] for i in group1_data if
                                i['template'] == template and i['emotion_word'] == emotion_word]
                group2_score = [i['score'] for i in group2_data if
                                i['template'] == template and i['emotion_word'] == emotion_word]
                if len(group1_score) > 0:
                    if np.mean(group1_score) > np.mean(group2_score):
                        diff1.append(np.mean(group1_score) - np.mean(group2_score))
                    else:
                        diff2.append(np.mean(group2_score) - np.mean(group1_score))
        print(f"{print_str}: avg diff1 is {np.mean(diff1)} and diff2 is {np.mean(diff2)}")
        return np.mean(diff1), np.mean(diff2)

    temp = [model_path[8:]]
    d1, d2 = comparision_between_two(new_eec, 'male', 'female', False, False, 'MF__')
    temp.append(d1)
    temp.append(d2)
    d1, d2 = comparision_between_two(new_eec, False, False, 'African-American', 'European', '__AE')
    temp.append(d1)
    temp.append(d2)
    d1, d2 = comparision_between_two(new_eec, 'male', 'male', 'African-American', 'European', 'MMAE')
    temp.append(d1)
    temp.append(d2)
    d1, d2 = comparision_between_two(new_eec, 'female', 'female', 'African-American', 'European', 'FFAE')
    temp.append(d1)
    temp.append(d2)
    d1, d2 = comparision_between_two(new_eec, 'male', 'female', 'African-American', 'African-American', 'MFAA')
    temp.append(d1)
    temp.append(d2)
    d1, d2 = comparision_between_two(new_eec, 'male', 'female', 'European', 'European', 'MFEE')
    temp.append(d1)
    temp.append(d2)
    d1, d2 = comparision_between_two(new_eec, 'male', 'female', 'African-American', 'European', 'MFAE')
    temp.append(d1)
    temp.append(d2)
    d1, d2 = comparision_between_two(new_eec, 'male', 'female', 'European', 'African-American', 'MFEA')
    temp.append(d1)
    temp.append(d2)
    print("****************")
    final_data.append(temp)
    t = Texttable()
    t.set_max_width(0)
    t.add_rows(final_data)
    print(t.draw())

t = Texttable()
t.set_max_width(0)
t.add_rows(final_data)
print(t.draw())