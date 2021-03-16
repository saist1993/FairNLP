# This file will replicate main.py but without torch.text experimental functionality.
# Secondary objective is to be as abstract and organized as possible

# python main.py -data bias_in_bios -is_regression False -clean_text True -tokenizer simple -use_pretrained_emb True -is_adv True -adv_loss_scale 0.5 -bs 1024 -embeddings ../../../storage/twitter_emb_200/simple_glove_vectors.vec
# Torch related imports
import torch
import torchtext
import torch.nn as nn
import torch.optim as optim


import click
import scipy
import random
import gensim
import pickle
from pathlib import Path
from tqdm.auto import tqdm
from typing import Optional, Callable, List


# custom imports
import create_data
import tokenizer_wrapper
from config import BILSTM_PARAMS
from utils import resolve_device, CustomError
from training_loop import basic_training_loop
from utils import clean_text as clean_text_function
from utils import clean_text_tweet as clean_text_function_tweet
from models import BiLSTM, initialize_parameters, BiLSTMAdv, BOWClassifier, Attacker

import bias_in_bios_analysis

def calculate_accuracy_classification(predictions, labels):
    top_predictions = predictions.argmax(1, keepdim = True)
    correct = top_predictions.eq(labels.view_as(top_predictions)).sum()
    accuracy = correct.float() / labels.shape[0]
    return accuracy

def calculate_accuracy_regression(predictions, labels):
    return scipy.stats.pearsonr(predictions.squeeze().detach().cpu().numpy(), labels.cpu().detach().cpu().numpy())[0]



def init_tokenizer(tokenizer:str,
                   clean_text:Optional[Callable],
                   max_length:Optional[int]):

    if tokenizer.lower() == 'spacy':
        return tokenizer_wrapper.SpacyTokenizer(spacy_model="en_core_web_sm", clean_text=clean_text, max_length=max_length)
    elif tokenizer.lower() == 'tweet':
        return tokenizer_wrapper.TwitterTokenizer(clean_text=clean_text_function_tweet, max_length=max_length)
    elif tokenizer.lower() == 'simple':
        return tokenizer_wrapper.SimpleTokenizer(clean_text=clean_text_function_tweet, max_length=max_length)
    else:
        raise CustomError("Tokenizer not found")

def generate_data_iterator(dataset_name:str, **kwargs):

    if dataset_name[:4].lower() == 'wiki':
        dataset_creator = create_data.WikiSimpleClassification(dataset_name=dataset_name,**kwargs)
        vocab, number_of_labels, train_iterator, dev_iterator, test_iterator = dataset_creator.run()
        return vocab, number_of_labels, train_iterator, dev_iterator, test_iterator

    elif dataset_name.lower() == 'valence':
        dataset_creator = create_data.ValencePrediction(dataset_name=dataset_name,**kwargs)
        vocab, number_of_labels, train_iterator, dev_iterator, test_iterator = dataset_creator.run()
        return vocab, number_of_labels, train_iterator, dev_iterator, test_iterator

    elif dataset_name.lower() == 'bias_in_bios':
        if kwargs['is_adv']:
            dataset_creator = create_data.BiasinBiosSimpleAdv(dataset_name=dataset_name, **kwargs)
            vocab, number_of_labels, train_iterator, dev_iterator, test_iterator = dataset_creator.run()
            return vocab, number_of_labels, train_iterator, dev_iterator, test_iterator
        else:
            dataset_creator = create_data.BiasinBiosSimple(dataset_name=dataset_name, **kwargs)
            vocab, number_of_labels, train_iterator, dev_iterator, test_iterator = dataset_creator.run()
            return vocab, number_of_labels, train_iterator, dev_iterator, test_iterator
    else:
        raise CustomError("No such dataset")

def get_pretrained_embedding(initial_embedding, pretrained_vectors, vocab, device):
    pretrained_embedding = torch.FloatTensor(initial_embedding.weight.clone()).cpu().detach().numpy()

    # if device == 'cpu':
    #     pretrained_embedding = torch.FloatTensor(initial_embedding.weight.clone()).cpu().detach().numpy()
    # else:
    #     pretrained_embedding = torch.FloatTensor(initial_embedding.weight.clone()).cuda().detach().numpy()

    unk_tokens = []

    for idx, token in tqdm(enumerate(vocab.itos)):
        try:
            pretrained_embedding[idx] = pretrained_vectors[token]
        except KeyError:
            unk_tokens.append(token)

    pretrained_embedding = torch.from_numpy(pretrained_embedding).to(device)
    return pretrained_embedding, unk_tokens


@click.command()
@click.option('-embedding', '--emb_dim', type=int, default=300)
@click.option('-spacy', '--spacy_model', type=str, default="en_core_web_sm", help="the spacy model used for tokenization. This might not be suitable for twitter and other noisy use cases ")
@click.option('-seed', '--seed', type=int, default=1234)
@click.option('-data', '--dataset_name', type=str, default='wiki_debias', help='the first half (wiki) is the name of the dataset, and the second half (debias) is the specific kind to use')
@click.option('-bs', '--batch_size', type=int, default=512)
@click.option('-pad', '--pad_token', type=str, default='<pad>')
@click.option('-unk', '--unk_token', type=str, default='<unk>')
@click.option('-embeddings', '--pre_trained_embeddings', type=str, default='../../bias-in-nlp/different_embeddings/simple_glove_vectors.vec') # work on this.
@click.option('-save_model_as', '--model_save_name', type=str, default='bilstm.pt')
@click.option('-model', '--model', type=str, default='bilstm')
@click.option('-is_regression', '--regression', type=bool, default=True, help='if regression then sentiment/toxicity is a continous value else classification.')
@click.option('-tokenizer', '--tokenizer_type', type=str, default="spacy", help='currently available: tweet, spacy, simple')
@click.option('-clean_text', '--use_clean_text', type=bool, default=False)
@click.option('-max_len', '--max_length', type=int, default=None)
@click.option('-epochs', '--epochs', type=int, default=30)
@click.option('-learnable_embeddings', '--learnable_embeddings', type=bool, default=False)
@click.option('-vocab_location', '--vocab_location', type=bool, default=False, help="file path location. Generally used while testing to load a vocab. Type is incorrect.")
@click.option('-is_adv', '--is_adv', type=bool, default=False, help="if True; adds an adversarial loss to the mix.")
@click.option('-adv_loss_scale', '--adv_loss_scale', type=float, default=0.5, help="sets the adverserial scale (lambda)")
@click.option('-use_pretrained_emb', '--use_pretrained_emb', type=bool, default=True, help="uses pretrianed if true else random")
@click.option('-default_emb_dim', '--default_emb_dim', type=int, default=100, help="uses pretrianed if true else random")
@click.option('-save_test_pred', '--save_test_pred', type=bool, default=False, help="has very specific use case: only works with adv_bias_in_bios")
@click.option('-noise_layer', '--noise_layer', type=bool, default=False, help="used for diff privacy. For now, not implemented")
@click.option('-eps', '--eps', type=float, default=1.0, help="privacy budget")
@click.option('-is_post_hoc', '--is_post_hoc', type=bool, default=False, help="trains a post-hoc classifier")
@click.option('-train_main_model', '--train_main_model', type=bool, default=True, help="If false; only trains post-hoc classifier")

def main(emb_dim:int,
         spacy_model:str,
         seed:int,
         dataset_name:str,
         batch_size:int,
         pad_token:str,
         unk_token:str,
         pre_trained_embeddings:str,
         model_save_name:str,
         model:str,
         regression:bool,
         tokenizer_type:str,
         use_clean_text:bool,
         max_length:Optional[int],
         epochs:int,
         learnable_embeddings:bool,
         vocab_location:Optional[None],
         is_adv:bool,
         adv_loss_scale:float,
         use_pretrained_emb:bool,
         default_emb_dim:int,
         save_test_pred:bool,
         noise_layer:bool,
         eps:float,
         is_post_hoc:bool,
         train_main_model:bool):

    print(f"seed is {seed}")
    torch.manual_seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    device = resolve_device() # if cuda: then cuda else cpu


    # set clean text, max length, vocab
    clean_text = clean_text_function if use_clean_text else None
    max_length = max_length if max_length else None
    vocab = pickle.load(open(vocab_location, 'rb')) if vocab_location else None

    print(f"initializing tokenizer: {tokenizer_type}")
    tokenizer = init_tokenizer(
        tokenizer= tokenizer_type,
        clean_text=clean_text,
        max_length=max_length
    )

    iterator_params = {
        'tokenizer': tokenizer,
        'artificial_populate': [],
        'pad_token': pad_token,
        'batch_size': batch_size,
        'is_regression': regression,
        'vocab': vocab,
        'is_adv': is_adv
    }
    vocab, number_of_labels, train_iterator, dev_iterator, test_iterator = \
        generate_data_iterator(dataset_name=dataset_name, **iterator_params)

    print(f"number of labels: {number_of_labels}")
    # need to pickle vocab. Same name as model save name but with additional "_vocab.pkl"
    pickle.dump(vocab, open(model_save_name + '_vocab.pkl', 'wb'))
    # load pre-trained embeddings
    if use_pretrained_emb:
        print(f"reading pre-trained vector file from: {pre_trained_embeddings}")
        pretrained_embedding = gensim.models.KeyedVectors.load_word2vec_format(pre_trained_embeddings)

        # infer input dim based on the pretrained_embeddings
        emb_dim = pretrained_embedding.vectors.shape[1]
    else:
        emb_dim = default_emb_dim

    output_dim = number_of_labels
    input_dim = len(vocab)

    if model == 'bilstm':
        model_params = {
            'input_dim': input_dim,
            'emb_dim': emb_dim,
            'hidden_dim': BILSTM_PARAMS['hidden_dim'],
            'output_dim': output_dim,
            'n_layers': BILSTM_PARAMS['n_layers'],
            'dropout': BILSTM_PARAMS['dropout'],
            'pad_idx': vocab['pad_token'],
            'adv_number_of_layers' : BILSTM_PARAMS['adv_number_of_layers'],
            'adv_dropout' : BILSTM_PARAMS['adv_dropout'],
            'device': device,
            'noise_layer': noise_layer,
            'eps': eps
        }
        if is_adv:
            model = BiLSTMAdv(model_params)
            model.apply(initialize_parameters)
        else:
            model = BiLSTM(model_params)
            model.apply(initialize_parameters)
    elif model == 'bow':
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
        model = BOWClassifier(model_params)
        model.apply(initialize_parameters)
    else:
        raise CustomError("No such model found")



    if use_pretrained_emb:
        print("updating embeddings")
        pretrained_embedding, unk_tokens = get_pretrained_embedding(initial_embedding=model.embedding,
                                                                    pretrained_vectors=pretrained_embedding,
                                                                    vocab=vocab,
                                                                    device=device)


        model.embedding.weight.data.copy_(pretrained_embedding)

    if not learnable_embeddings:
        model.embedding.weight.requires_grad = False

    print("model initialized")
    model = model.to(device)

    # setting up optimizer
    optimizer = optim.Adam(model.parameters([param for param in model.parameters() if param.requires_grad == True]), lr=0.01)

    # setting up loss function
    if number_of_labels == 1:
        criterion = nn.MSELoss()
    else:
        criterion = nn.CrossEntropyLoss()

    # Things still left
    accuracy_calculation_function = calculate_accuracy_regression if number_of_labels == 1 else calculate_accuracy_classification



    if train_main_model:
        other_params = {
            'is_adv': is_adv,
            'loss_aux_scale': adv_loss_scale,
            'is_regression': regression,
            'is_post_hoc': False # here the post-hoc has to be false
        }

        best_test_acc, best_valid_acc, test_acc_at_best_valid_acc = basic_training_loop(
             n_epochs=epochs,
             model=model,
             train_iterator=train_iterator,
             dev_iterator=dev_iterator,
             test_iterator=test_iterator,
             optimizer=optimizer,
             criterion=criterion,
             device=device,
             model_save_name=model_save_name,
             accuracy_calculation_function = accuracy_calculation_function,
             other_params=other_params
        )

        print(f"BEST Test Acc: {best_test_acc} || Actual Test Acc: {test_acc_at_best_valid_acc} || Best Valid Acc {best_valid_acc}")


    if is_post_hoc:
        # the post_hoc classifier will be trained.
        assert is_adv == True
        model_params['return_hidden'] = True

        # step 1 -> load the main model
        if is_adv:
            model = BiLSTMAdv(model_params)
        else:
            model = BiLSTM(model_params)
        model.load_state_dict(torch.load(model_save_name))
        model = model.to(device)

        # step 2 -> init the post-hoc model
        post_hoc = Attacker(model_params,model)
        post_hoc = post_hoc.to(device)
        optimizer = optim.Adam(model.parameters([param for param in post_hoc.parameters() if param.requires_grad == True]),
                               lr=0.01)


        # step 4 -> train like a normal human

        other_params = {
            'is_adv': is_adv,
            'loss_aux_scale': adv_loss_scale,
            'is_regression': regression,
            'is_post_hoc': True # here the post-hoc has to be false
        }

        best_test_acc, best_valid_acc, test_acc_at_best_valid_acc = basic_training_loop(
             n_epochs=epochs,
             model=post_hoc,
             train_iterator=train_iterator,
             dev_iterator=dev_iterator,
             test_iterator=test_iterator,
             optimizer=optimizer,
             criterion=criterion,
             device=device,
             model_save_name=model_save_name + 'post_hoc.pt',
             accuracy_calculation_function = accuracy_calculation_function,
             other_params=other_params
        )

        print(f"BEST Test Acc for post hoc: {best_test_acc} || Actual Test Acc: {test_acc_at_best_valid_acc} || Best Valid Acc {best_valid_acc}")


    if save_test_pred:
        print("running experiments over test pred: Only valid in specific conditions")
        if is_adv:
            model = BiLSTMAdv(model_params)
        else:
            model = BiLSTM(model_params)
        model.load_state_dict(torch.load(model_save_name))
        model = model.to(device)

        test_data = pickle.load(open("../data/bias_in_bios/test.pickle", "rb"))
        profession_to_id = pickle.load(open("../data/bias_in_bios/profession_to_id.pickle","rb"))
        id_to_profession = {id:prof for prof, id in profession_to_id.items()}

        bias_in_bios_analysis.generate_predictions(
            model=model,
            data=test_data,
            id_to_profession=id_to_profession,
            tokenizer=tokenizer,
            vocab=vocab,
            device=device,
            save_data_at=model_save_name+ '_test_pred.pkl'
        )


if __name__ == '__main__':
    main()

