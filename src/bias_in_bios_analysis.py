# Provides some functions which can be called to provide some basic analysis


import spacy
import pickle
import numpy as np


from main import *
from utils import resolve_device
from utils import clean_text as clean_text_function

from config import BILSTM_PARAMS
from models import BiLSTM, BiLSTMAdv







def predict(tokenizer, vocab, model, device, sentence):
    model.eval()
    tokens = tokenizer.tokenize(sentence)
    length = torch.LongTensor([len(tokens)]) # length is on CPU
    indexes = [vocab.stoi[token] for token in tokens]
    tensor = torch.LongTensor(indexes).unsqueeze(-1).to(device)
    prediction = model(tensor, length)[0]
    probabilities = nn.functional.softmax(prediction, dim = -1)
    return torch.argmax(probabilities.squeeze()).item()


def get_acc(sub_data:list):
    acc = []
    for t in sub_data:
        if t['p'] == t['pred_prof']:
            acc.append(1.0)
        else:
            acc.append(0.0)
    return np.mean(acc)

def calculate_rms(acc_data):
    acc = [(i[2]-i[3])**2 for i in acc_data]
    return np.sqrt(np.sum(acc))

def return_larget_diff(acc_data):
    profession, diff = acc_data[0][0], abs(acc_data[0][1]-acc_data[0][2])
    for i in acc_data[1:]:
        if diff < abs(i[1]-i[2]):
            profession, diff = i[0], abs(i[1]-i[2])
    return profession, diff


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
    print(f"device here is {device}")
    new_data = []
    for t in tqdm(data):
        pred_prof = id_to_profession[
            predict(tokenizer=tokenizer, vocab=vocab, model=model, device=device, sentence=t['hard_text_untokenized'])]
        t['pred_prof'] = pred_prof
        new_data.append(t)

    pickle.dump(new_data, open(save_data_at, 'wb'))

    test_examples_professional = {prof: {'male': [], 'female': []} for id, prof in id_to_profession.items()}

    for data_point in new_data:

        if data_point['g'] == 'f':
            test_examples_professional[data_point['p']]['female'] = test_examples_professional[data_point['p']][
                                                                        'female'] + [data_point]
        elif data_point['g'] == 'm':
            test_examples_professional[data_point['p']]['male'] = test_examples_professional[data_point['p']][
                                                                      'male'] + [data_point]
        else:
            raise KeyError

    final_acc = []
    for key, value in test_examples_professional.items():
        temp = [
            key,
            get_acc(value['male'] + value['female']),
            get_acc(value['male']),
            get_acc(value['female'])
        ]
        final_acc.append(temp)

    print(final_acc)
    print(calculate_rms(final_acc))
    print(return_larget_diff(final_acc))





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
    print(f"current device is {device}")

    model_path = '../../../storage/model_toxic/bilstm_adv.pt' # '../toxic_models/bilstm_adv.pt'
    vocab_path = '../../../storage/model_toxic/bilstm_adv.pt_vocab.pkl'
    prof_to_id = '../data/bias_in_bios/profession_to_id.pickle'

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
    emb_dim = 100
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
