"""From mytorch utils"""

import re
import warnings
import numpy as np
from tqdm import tqdm
from typing import List, Dict

import torch
import torch.nn as nn
from torch.autograd import Function


class CustomError(Exception):
    pass

class ImproperCMDArguments(Exception): pass

# Transparent, and simple argument parsing FTW!
def convert_nicely(arg, possible_types=(bool, float, int, str)):
    """ Try and see what sticks. Possible types can be changed. """
    for data_type in possible_types:
        try:

            if data_type is bool:
                # Hard code this shit
                if arg in ['T', 'True', 'true']: return True
                if arg in ['F', 'False', 'false']: return False
                raise ValueError
            else:
                proper_arg = data_type(arg)
                return proper_arg
        except ValueError:
            continue
    # Here, i.e. no data type really stuck
    warnings.warn(f"None of the possible datatypes matched for {arg}. Returning as-is")
    return arg


def parse_args(raw_args: List[str], compulsory: List[str] = (), compulsory_msg: str = "",
               types: Dict[str, type] = None, discard_unspecified: bool = False):
    """
        I don't like argparse.
        Don't like specifying a complex two liner for each every config flag/macro.
        If you maintain a dict of default arguments, and want to just overwrite it based on command args,
        call this function, specify some stuff like
    :param raw_args: unparsed sys.argv[1:]
    :param compulsory: if some flags must be there
    :param compulsory_msg: what if some compulsory flags weren't there
    :param types: a dict of confignm: type(configvl)
    :param discard_unspecified: flag so that if something doesn't appear in config it is not returned.
    :return:
    """

    # parsed_args = _parse_args_(raw_args, compulsory=compulsory, compulsory_msg=compulsory_msg)
    #
    # # Change the "type" of arg, anyway

    parsed = {}

    while True:

        try:                                        # Get next value
            nm = raw_args.pop(0)
        except IndexError:                          # We emptied the list
            break

        # Get value
        try:
            vl = raw_args.pop(0)
        except IndexError:
            raise ImproperCMDArguments(f"A value was expected for {nm} parameter. Not found.")

        # Get type of value
        if types:
            try:
                parsed[nm] = types[nm](vl)
            except ValueError:
                raise ImproperCMDArguments(f"The value for {nm}: {vl} can not take the type {types[nm]}! ")
            except KeyError:                    # This name was not included in the types dict
                if not discard_unspecified:     # Add it nonetheless
                    parsed[nm] = convert_nicely(vl)
                else:                           # Discard it.
                    continue
        else:
            parsed[nm] = convert_nicely(vl)

    # Check if all the compulsory things are in here.
    for key in compulsory:
        try:
            assert key in parsed
        except AssertionError:
            raise ImproperCMDArguments(compulsory_msg + f"Found keys include {[k for k in parsed.keys()]}")

    # Finally check if something unwanted persists here
    return parsed


def clean_text(text:str):
    """
    cleans text casing puntations and special characters. Removes extra space
    """
    text = re.sub('[^ a-zA-Z0-9]|unk', '', text)
    text = text.strip()
    return text

def clean_text_tweet(text:str):
    return text.replace('#', '').replace('@', '')

def get_pretrained_embedding(initial_embedding, pretrained_vocab, pretrained_vectors, vocab, unk_token, device):
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


def resolve_device(device = None) -> torch.device:
    """Resolve a torch.device given a desired device (string)."""
    if device is None or device == 'gpu':
        device = 'cuda'
    if isinstance(device, str):
        device = torch.device(device)
    if not torch.cuda.is_available() and device.type == 'cuda':
        device = torch.device('cpu')
        print('No cuda devices were available. The model runs on CPU')
    return device


def laplace(epsilon, L1_norm):
    b = L1_norm / epsilon
    return b

import torch
from torchtext.data.utils import ngrams_iterator

__all__ = [
    'vocab_func',
    'totensor',
    'ngrams_func',
    'sequential_transforms'
]


def vocab_func(vocab):
    def func(tok_iter):
        return [vocab[tok] for tok in tok_iter]

    return func


def totensor(dtype):
    def func(ids_list):
        return torch.tensor(ids_list).to(dtype)

    return func


def ngrams_func(ngrams):
    def func(token_list):
        return list(ngrams_iterator(token_list, ngrams))

    return func


def sequential_transforms(*transforms):
    def func(txt_input):
        for transform in transforms:
            txt_input = transform(txt_input)
        return txt_input

    return func


class GradReverse(Function):
    """
        Torch function used to invert the sign of gradients (to be used for argmax instead of argmin)
        Usage:
            x = GradReverse.apply(x) where x is a tensor with grads.

        Copied from here: https://github.com/geraltofrivia/mytorch/blob/0ce7b23ff5381803698f6ca25bad1783d21afd1f/src/mytorch/utils/goodies.py#L39
    """
    @staticmethod
    def forward(ctx, x):
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg()


class TextClassificationDataset(torch.utils.data.Dataset):
    """Defines an abstract text classification datasets.
       Currently, we only support the following datasets:

           - AG_NEWS
           - SogouNews
           - DBpedia
           - YelpReviewPolarity
           - YelpReviewFull
           - YahooAnswers
           - AmazonReviewPolarity
           - AmazonReviewFull
    """

    def __init__(self, data, vocab, transforms):
        """Initiate text-classification dataset.

        Assumption is that the first element in the list i.e. 0th is the label
        Args:
            data: a list of label and text tring tuple. label is an integer.
                [(label1, text1), (label2, text2), (label2, text3)]
            vocab: Vocabulary object used for dataset.
            transforms: a tuple of label and text string transforms.
        """

        super(TextClassificationDataset, self).__init__()
        self.data = data
        self.vocab = vocab
        self.transforms = transforms  # (label_transforms, tokens_transforms)
        # can I do some static mechanism to find all y and g;s

    def __getitem__(self, i):
        label = self.data[i][0]
        txt = self.data[i][1]
        # return (self.transforms[0](label), self.transforms[1](txt))

        final_data = []
        for data, transformation in zip(self.data[i], self.transforms):
            final_data.append(transformation(data))

        return tuple(final_data)

    def __len__(self):
        return len(self.data)

    def get_labels(self):
        labels = []
        for item in self.data:
            label = item[0]
            labels.append(self.transforms[0](label))
        return set(labels)

    def get_vocab(self):
        return self.vocab


def get_enc_grad_norm(model):


        tn = 0

        for p in model.embedder.parameters(2):
            try:
                pn = p.grad.data.norm(2)
                tn += pn.item() ** 2
            except:
                continue

        tn = tn ** (1. / 2)
        return tn


def equal_odds(preds, y, s, device, total_no_main_classes, total_no_aux_classes, epsilon=0.0):
    """

    :param preds: output/prediction of the model
    :param y: actual/ground/gold label
    :param s: aux output/ protected demographic attribute
    :param epsilon:
    :return:
    """

    unique_classes = torch.sort(torch.unique(y))[0] # For example: [doctor, nurse, engineer]
    fairness = torch.zeros(s.shape).to(device)
    unique_groups = torch.sort(torch.unique(s))[0] # For example: [Male, Female]
    group_fairness = {} # a dict which keeps a track on how fairness is changing
    fairness_lookup = torch.zeros([total_no_main_classes, total_no_aux_classes])
    '''
    it will have a structure of 
    {
        'doctor': {
            'm' : 0.5, 
            'f' : 0.6
            }, 
        'model': {
        'm': 0.5,
        'f': 0.7,
        }
    '''
    for uc in unique_classes: # iterating over each class say: uc=doctor for the first iteration
        group_fairness[uc] = {}
        positive_rate = torch.mean((preds[y==uc] == uc).float()) # prob(pred=doctor/y=doctor)
        for group in unique_groups: # iterating over each group say: group=male for the firt iteration
            mask_pos = torch.logical_and(y==uc, s==group) # find instances with y=doctor and s=male
            g_fairness_pos = torch.mean((preds[mask_pos] == uc).float()) - positive_rate
            g_fairness_pos = torch.sign(g_fairness_pos) * torch.clip(torch.abs(g_fairness_pos) - epsilon, 0, None)
            fairness[mask_pos] = g_fairness_pos
            group_fairness[uc][group] = g_fairness_pos
            fairness_lookup[int(uc.item()),int(group.item())] = g_fairness_pos

    return group_fairness, fairness_lookup

def calculate_grms(preds, y, s):
    unique_classes = torch.sort(torch.unique(y))[0]  # For example: [doctor, nurse, engineer]
    unique_groups = torch.sort(torch.unique(s))[0]  # For example: [Male, Female]
    group_fairness = {}  # a dict which keeps a track on how fairness is changing

    '''
    it will have a structure of 
    {
        'doctor': {
            'm' : 0.5, 
            'f' : 0.6
            }, 
        'model': {
        'm': 0.5,
        'f': 0.7,
        }
    '''
    for uc in unique_classes:  # iterating over each class say: uc=doctor for the first iteration
        group_fairness[uc] = {}
        positive_rate = torch.mean((preds[y == uc] == uc).float())  # prob(pred=doctor/y=doctor)
        for group in unique_groups:  # iterating over each group say: group=male for the firt iteration
            mask_pos = torch.logical_and(y == uc, s == group)  # find instances with y=doctor and s=male
            g_fairness_pos = torch.mean((preds[mask_pos] == uc).float())
            print(g_fairness_pos)
            group_fairness[uc][group] = g_fairness_pos.item()

    scores = []
    for key,value in group_fairness.items():
        temp = [value_1 for key_1, value_1 in value.items()]
        gender_1, gender_2 = temp[0], temp[1]
        scores.append((gender_1-gender_2)**2)

    print(group_fairness)
    return np.sqrt(np.mean(scores))