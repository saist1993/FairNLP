# Provides some functions which can be called to provide some basic analysis

import torch
import torch.nn as nn


from tqdm.auto import tqdm
import pickle

from sklearn.metrics import roc_auc_score, auc
from string import Template
from tqdm.auto import tqdm
import spacy
import itertools

from main import *
from utils import resolve_device
from utils import clean_text as clean_text_function
from utils import clean_text_tweet as clean_text_function_tweet

from config import BILSTM_PARAMS
from models import BiLSTM, BiLSTMAdv


import pickle
from pathlib import Path
from utils import GradReverse



def predict(tokenizer, vocab, model, device, sentence):
    model.eval()
    tokens = tokenizer.tokenize(sentence)
    length = torch.LongTensor([len(tokens)]).to(device)
    indexes = [vocab.stoi[token] for token in tokens]
    tensor = torch.LongTensor(indexes).unsqueeze(-1).to(device)
    prediction = model(tensor, length)[0]
    probabilities = nn.functional.softmax(prediction, dim = -1)
    return torch.argmax(probabilities.squeeze()).item()



def generate_predictions(model,
                         data,
                         id_to_profession,
                         tokenizer,
                         vocab,
                         device,
                         save_data_at):
    """
    Runs the model over the data and return a set
    :param model:
    :param data:
    :param id_to_professional:
    :return:
    """

    new_data = []
    for t in tqdm(data):
        pred_prof = id_to_profession[
            predict(tokenizer=tokenizer, vocab=vocab, model=model, device=device, sentence=t['hard_text_untokenized'])]
        t['pred_prof'] = pred_prof
        new_data.append(t)

    pickle.dump(new_data, open(save_data_at, 'wb'))

if __name__ == '__main__':
    def load_dataset(path):
        with open(path, "rb") as f:
            data = pickle.load(f)
        return data


    train = load_dataset("../data/bias_in_bios/train.pickle")
    dev = load_dataset("../data/bias_in_bios/dev.pickle")
    test = load_dataset("../data/bias_in_bios/test.pickle")

    seed = 1234

    torch.manual_seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    nlp = spacy.load("en_core_web_sm")
    device = resolve_device()

    model_path = '../toxic_models/bilstm_adv.pt'
    vocab_path = '../toxic_models/bilstm_adv.pt_vocab.pkl'
    prof_to_id = '../toxic_models/profession_to_id.pickle'

    prof_to_id = pickle.load(open(prof_to_id, 'rb'))
    id_to_prof = {value: key for key, value in prof_to_id.items()}

    # params which will change
    tokenizer = 'simple'
    clean_text = clean_text_function
    regression = False
    dataset_name = 'wiki_debias'

    # mostly static
    max_length = None
    pad_token = '<pad>'
    batch_size = 512

    # model params
    emb_dim = 300
    is_adv = True

    tokenizer = init_tokenizer(tokenizer=tokenizer,
                               clean_text=clean_text,
                               max_length=None)

    iterator_params = {
        'tokenizer': tokenizer,
        'artificial_populate': [],
        'pad_token': pad_token,
        'batch_size': batch_size,
        'is_regression': regression,
        'vocab': pickle.load(open(vocab_path, 'rb')),
        'is_adv': is_adv
    }

    vocab, number_of_labels, train_iterator, dev_iterator, test_iterator = \
        generate_data_iterator(dataset_name=dataset_name, **iterator_params)

    input_dim = len(vocab)
    all_profession = list(set([t['p'] for t in train]))
    output_dim = len(all_profession)

    model_params = {
        'input_dim': input_dim,
        'emb_dim': emb_dim,
        'hidden_dim': BILSTM_PARAMS['hidden_dim'],
        'output_dim': output_dim,
        'n_layers': BILSTM_PARAMS['n_layers'],
        'dropout': BILSTM_PARAMS['dropout'],
        'pad_idx': vocab['pad_token'],
        'adv_number_of_layers': BILSTM_PARAMS['adv_number_of_layers'],
        'adv_dropout': BILSTM_PARAMS['adv_dropout'],
        'device': device
    }

    model = BiLSTMAdv(model_params)
    model.load_state_dict(torch.load(model_path, map_location=torch.device(device)))

    if number_of_labels == 1:
        criterion = nn.MSELoss()
    else:
        criterion = nn.CrossEntropyLoss()

    optimizer = optim.Adam(model.parameters([param for param in model.parameters() if param.requires_grad == True]))

    generate_predictions(model=model,
                         data=test,
                         id_to_profession=id_to_prof,
                         tokenizer=tokenizer,
                         vocab=vocab,
                         device=device,
                         save_data_at='../toxic_models/bilstm_adv.pt.test_pred.pkl')
