"""From mytorch utils"""

import re
import math
import warnings
import numpy as np
from tqdm import tqdm
from typing import List, Dict
from sklearn.metrics import confusion_matrix

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


def demographic_parity(preds, y, s, device, total_no_main_classes, total_no_aux_classes, epsilon=0.0):
    """
    We assume y to be 0,1  that is y can only take 1 or 0 as the input.

    y = 1 is considered to be a positive class while y=0 is considered to be a negative class. This is
    unlike other methods where y=-1 is considered to be negative class.

    In demographic parity the fairness score for class y=0 is 0.

    """

    unique_classes = torch.sort(torch.unique(y))[0] # For example: [doctor, nurse, engineer]
    assert total_no_main_classes == 2 # only valid for two classes where the negative class is assumed to be y=0
    assert len(unique_classes) <= 2 # as there can't be more than 2 classes

    fairness = torch.zeros(s.shape).to(device)
    unique_groups = torch.sort(torch.unique(s))[0] # For example: [Male, Female]
    group_fairness = {} # a dict which keeps a track on how fairness is changing
    fairness_lookup = torch.zeros([total_no_main_classes, total_no_aux_classes])

    positive_rate = torch.mean((preds == 1).float())  # prob(pred=doctor/y=doctor)

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

        for group in unique_groups: # iterating over each group say: group=male for the firt iteration
            mask_pos = (s == group)
            if uc == 1:
                 # find instances with y=doctor and s=male
                g_fairness_pos = torch.mean((preds[mask_pos] == uc).float()) - positive_rate
                g_fairness_pos = torch.sign(g_fairness_pos) * torch.clip(torch.abs(g_fairness_pos) - epsilon, 0, None)
            else: # uc = 0 which in our case is negative class
                g_fairness_pos = torch.tensor(0.0).to(device) # TODO: check if g_fairness_pos in the obove if condition is of the same type
            fairness[mask_pos] = g_fairness_pos
            group_fairness[uc][group] = g_fairness_pos
            fairness_lookup[int(uc.item()),int(group.item())] = g_fairness_pos


    return group_fairness, fairness_lookup


def equal_opportunity(preds, y, s, device, total_no_main_classes, total_no_aux_classes, epsilon=0.0):
    """
    We assume y to be 0,1  that is y can only take 1 or 0 as the input.

    y = 1 is considered to be a positive class while y=0 is considered to be a negative class. This is
    unlike other methods where y=-1 is considered to be negative class.

    In demographic parity the fairness score for class y=0 is 0.

    """

    unique_classes = torch.sort(torch.unique(y))[0] # For example: [doctor, nurse, engineer]
    assert total_no_main_classes == 2 # only valid for two classes where the negative class is assumed to be y=0
    assert len(unique_classes) <= 2 # as there can't be more than 2 classes

    fairness = torch.zeros(s.shape).to(device)
    unique_groups = torch.sort(torch.unique(s))[0] # For example: [Male, Female]
    group_fairness = {} # a dict which keeps a track on how fairness is changing
    fairness_lookup = torch.zeros([total_no_main_classes, total_no_aux_classes])

    positive_rate = torch.mean((preds[y == 1] == 1).float())  # prob(pred=doctor/y=doctor)

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

        for group in unique_groups: # iterating over each group say: group=male for the firt iteration
            mask_pos = torch.logical_and(y == uc, s == group)
            if uc == 1:
                 # find instances with y=doctor and s=male
                g_fairness_pos = torch.mean((preds[mask_pos] == uc).float()) - positive_rate
                g_fairness_pos = torch.sign(g_fairness_pos) * torch.clip(torch.abs(g_fairness_pos) - epsilon, 0, None)
            else: # uc = 0 which in our case is negative class
                g_fairness_pos = torch.tensor(0.0).to(device) # TODO: check if g_fairness_pos in the obove if condition is of the same type
            fairness[mask_pos] = g_fairness_pos # imposing the scores on the mask.
            group_fairness[uc][group] = g_fairness_pos
            fairness_lookup[int(uc.item()),int(group.item())] = g_fairness_pos


    return group_fairness, fairness_lookup


def calculate_demographic_parity(preds,y,s, other_params):
    """assumes two classes with positive class as y=1"""
    device = other_params['device']
    total_no_aux_classes = other_params['total_no_aux_classes']
    total_no_main_classes = other_params['total_no_main_classes']
    group_fairness, fairness_lookup = demographic_parity(preds, y, s, device, total_no_main_classes, total_no_aux_classes, epsilon=0.0)
    scores = []
    for key, value in group_fairness.items():
        if int(key.item()) == 1:
            for key1, value1 in value.items():
                scores.append(value1.item())
    return scores, group_fairness


def calculate_equal_opportunity(preds,y,s, other_params):
    """assumes two classes with positive class as y=1"""
    device = other_params['device']
    total_no_aux_classes = other_params['total_no_aux_classes']
    total_no_main_classes = other_params['total_no_main_classes']
    group_fairness, fairness_lookup = equal_opportunity(preds, y, s, device, total_no_main_classes, total_no_aux_classes, epsilon=0.0)
    scores = []
    for key, value in group_fairness.items():
        if int(key.item()) == 1:
            for key1, value1 in value.items():
                scores.append(value1.item())
    return scores, group_fairness


def calculate_equal_odds(preds,y,s, other_params):
    """assumes two classes with positive class as y=1"""
    device = other_params['device']
    total_no_aux_classes = other_params['total_no_aux_classes']
    total_no_main_classes = other_params['total_no_main_classes']
    group_fairness, fairness_lookup = equal_odds(preds, y, s, device, total_no_main_classes, total_no_aux_classes, epsilon=0.0)
    sorted_class = np.sort([key.item() for key,value in group_fairness.items()])
    group_fairness = {key.item():value for key,value in group_fairness.items()}
    scores = []
    for key in sorted_class:
        for key1, value1 in group_fairness[key].items():
            scores.append(value1.item())
    return scores, group_fairness


def calculate_grms(preds, y, s, other_params=None):
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
            temp = g_fairness_pos.item()
            if math.isnan(temp):
                temp = 0.0
            group_fairness[uc][group] = temp
        group_fairness[uc]['total_acc'] = positive_rate

    scores = []
    for key,value in group_fairness.items():
        temp = [value_1 for key_1, value_1 in value.items()]
        gender_1, gender_2 = temp[0], temp[1]
        scores.append((gender_1-gender_2)**2)

    return [np.sqrt(np.mean(scores))], group_fairness


def calculate_acc_diff(preds, y, s, other_params=None):
    # unique_classes = torch.sort(torch.unique(y))[0]  # For example: [doctor, nurse, engineer]
    all_acc= []
    main_acc = torch.mean((preds == y).float())
    all_acc.append(main_acc)

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
    # for uc in unique_classes:  # iterating over each class say: uc=doctor for the first iteration
    #     group_fairness[uc] = {}
    #     positive_rate = torch.mean((preds[y == uc] == uc).float())  # prob(pred=doctor/y=doctor)
    for group in unique_groups:  # iterating over each group say: group=male for the firt iteration
        mask_pos = s == group  # find instances with y=doctor and s=male
        acc_class = torch.mean((preds[mask_pos] == y[mask_pos]).float())
        temp = acc_class.item()
        if math.isnan(temp):
            temp = 0.0
        all_acc.append(temp)

    scores = abs(all_acc[1] - all_acc[2])
    assert len(all_acc) == 3 # there are just two groups.

    return [scores], all_acc

def calculate_true_rates(preds, y, s, other_params):
    '''
    inspired from https://github.com/HanXudong/Diverse_Adversaries_for_Mitigating_Bias_in_Training/blob/b5b4c99ada17b3c19ab2ae8789bb56058cb72643/networks/eval_metrices.py#L14
    :param preds:
    :param y:
    :param s:
    :param other_params:
    :return:
    '''
    unique_group = torch.sort(torch.unique(s))[0]
    assert len(unique_group) == 2

    g1_preds = preds[s == 1]
    g1_labels = y[s == 1]

    g0_preds = preds[s == 0]
    g0_labels = y[s == 0]


    tn0, fp0, fn0, tp0 = confusion_matrix(g0_labels.cpu().detach().numpy(), g0_preds.cpu().detach().numpy()).ravel()
    TPR0 = tp0/(fn0+tp0)
    TNR0 = tn0/(fp0+tn0)

    tn1, fp1, fn1, tp1 = confusion_matrix(g1_labels.cpu().detach().numpy(), g1_preds.cpu().detach().numpy()).ravel()
    TPR1 = tp1/(fn1+tp1)
    TNR1 = tn1/(tn1+fp1)

    TPR_gap = TPR0-TPR1
    TNR_gap = TNR0-TNR1

    return [TPR_gap,TNR_gap], ((abs(TPR_gap)+abs(TNR_gap))/2.0)


def custom_equal_odds(preds, y, s, device, total_no_main_classes, total_no_aux_classes, epsilon=0.0):
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


def calculate_ddp_dde(preds, y, s, other_params):
    preds = preds.detach().cpu().numpy()
    y = y.detach().cpu().numpy()
    s = s.detach().cpu().numpy()
    positive_rate_prot = get_positive_rate(preds[s == 0], y[s == 0])
    positive_rate_unprot = get_positive_rate(preds[s == 1], y[s == 1])
    true_positive_rate_prot = get_true_positive_rate(preds[s == 0], y[s == 0])
    true_positive_rate_unprot = get_true_positive_rate(preds[s == 1], y[s == 1])
    DDP = positive_rate_unprot - positive_rate_prot
    DEO = true_positive_rate_unprot - true_positive_rate_prot

    return [DDP, DEO], ((abs(DDP)+abs(DEO))/2.0)


def get_positive_rate(y_predicted, y_true):
    """Compute the positive rate for given predictions of the class label.
    Parameters
    ----------
    y_predicted: numpy array
        The predicted class labels of shape=(number_points,).
    y_true: numpy array
        The true class labels of shape=(number_points,).
    Returns
    ---------
    pr: float
        The positive rate.
    """
    tn, fp, fn, tp = confusion_matrix(y_true, y_predicted).ravel()
    pr = (tp+fp) / (tp+fp+tn+fn)
    return pr

def get_true_positive_rate(y_predicted, y_true):
    """Compute the true positive rate for given predictions of the class label.
    Parameters
    ----------
    y_predicted: numpy array
        The predicted class labels of shape=(number_points,).
    y_true: numpy array
        The true class labels of shape=(number_points,).
    Returns
    ---------
    tpr: float
        The true positive rate.
    """
    tn, fp, fn, tp = confusion_matrix(y_true, y_predicted).ravel()
    tpr = tp / (tp+fn)
    return tpr


def calculate_multiple_things(preds, y, s, other_params):
    """
    do something here. Find ones which are useful and bunch them together to form easy dict
    """
    grms_score, grms_group_fairness = calculate_grms(preds, y, s, other_params)
    acc_diff_scores, acc_diff_all_acc = calculate_acc_diff(preds, y, s, other_params)
    [ddp,deo], _ = calculate_ddp_dde(preds, y, s, other_params)
    [tpr_gap, tnr_gap], avg_tpr_tnr = calculate_true_rates(preds, y, s, other_params)

    scores = {
        'grms': grms_score,
        'acc_diff': acc_diff_scores,
        'ddp': ddp,
        'deo': deo,
        'tpr_gap': tpr_gap,
        'tnr_gap': tnr_gap,
        'avg_tpr_tnr': avg_tpr_tnr
    }

    return grms_score, scores


def calculate_multiple_things_blog(preds, y, s, other_params):
    """
    do something here. Find ones which are useful and bunch them together to form easy dict
    """
    grms_score, grms_group_fairness = calculate_grms(preds, y, s, other_params)
    acc_diff_scores, acc_diff_all_acc = calculate_acc_diff(preds, y, s, other_params)

    scores = {
        'grms': grms_score,
        'acc_diff': acc_diff_scores
    }

    return grms_score, scores


if __name__ == '__main__':



    print(a)