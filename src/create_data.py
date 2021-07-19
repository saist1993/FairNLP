# reads, cleans, tokenizes data and create data specific iterators, and vocabulary

import torch
import pickle
import random
import torchtext
import numpy as np
import collections
import pandas as pd
import torch.nn as nn
from pathlib import Path
from string import Template
from typing import List, Callable
import dataset_reader_helper as drh
from sklearn.datasets import load_svmlight_files

torch.manual_seed(1234)
torch.cuda.manual_seed(1234)
torch.cuda.manual_seed_all(1234)
random.seed(1234)
np.random.seed(1234)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

from utils import totensor, sequential_transforms, vocab_func, \
    TextClassificationDataset, clean_text_tweet, CombinedIterator

class WikiSimpleClassification:

    def __init__(self, dataset_name:str,**params):
        '''
        reads, cleans, tokenizes data and create data specific iterators, and vocabulary for wiki (toxicity) tasks.
        More info here:
        '''
        self.dataset_name = dataset_name
        self.tokenizer = params['tokenizer']
        try:
            self.type = dataset_name.split('_')[1]  # type can be debias, main etc.
        except:
            self.type = None
        self.artificial_populate = params['artificial_populate']
        self.pad_token = params['pad_token']
        self.batch_size = params['batch_size']
        self.is_regression = params['is_regression']
        self.pad_idx = -1 # this gets updated in the run function once the vocab is made.
        self.vocab = params['vocab'] # while training set this to be None
        try:
            self.seed = params['seed']
        except:
            self.seed = 1234

        try:
            self.trim_data = params['trim_data']
        except KeyError:
            self.trim_data = False

    def transform_dataframe_to_dict(self, data_frame:pd, tokenizer:Callable):
        """

        reads the actual data from the disk
        @ still need to make it more general, specifcally path needs to be read from some external file.
        """
        print(tokenizer)
        final_data = []
        for i in data_frame.iterrows():
            temp = {
                'lable': i[1]['is_toxic'],
                'toxicity': i[1]['toxicity'],
                'text': i[1]['comment']
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


    def build_vocab_from_data(self, raw_train_data:List, raw_dev_data:List, artificial_populate:List):
        """This has been made customly for the given dataset. Need to write your own for any other use case"""

        token_freqs = collections.Counter()
        for data_point in raw_train_data:
            token_freqs.update(data_point['tokenized_text'])
        for data_point in raw_dev_data:
            token_freqs.update(data_point['tokenized_text'])
        #     token_freqs.update(data_point['tokenized_text'] for data_point in raw_train_data)
        #     token_freqs.update(data_point['tokenized_text'] for data_point in raw_dev_data)
        if artificial_populate:
            token_freqs.update(artificial_populate)
        vocab = torchtext.vocab.Vocab(token_freqs)
        return vocab

    def process_data(self, raw_data, vocab):
        """raw data is assumed to be tokenized"""
        final_data = [(data_point['lable'], data_point['tokenized_text']) for data_point in raw_data]
        text_transformation = sequential_transforms(vocab_func(vocab),
                                                    totensor(dtype=torch.long))
        if self.is_regression:
            label_transform = sequential_transforms(totensor(dtype=torch.float))
        else:
            label_transform = sequential_transforms(totensor(dtype=torch.long))


        transforms = (label_transform, text_transformation)

        return TextClassificationDataset(final_data, vocab, transforms)

    def collate(self, batch):
        labels, text = zip(*batch)
        if self.is_regression:
            labels = torch.FloatTensor(labels)
        else:
            labels = torch.LongTensor(labels)
        lengths = torch.LongTensor([len(x) for x in text])

        text = nn.utils.rnn.pad_sequence(text, padding_value=self.pad_idx)

        return labels, text, lengths

    def run(self):


        print("reading and tokenizing data. This is going to take long time: 10 mins ")
        # Optimize this later. We don't need pandas dataframe

        file = Path(f"../data/wiki/wiki_{self.type}_train.pkl")

        if file.exists():
            debias_train_raw = pickle.load(open(f'../data/wiki/wiki_{self.type}_train.pkl', 'rb'))
            debias_dev_raw = pickle.load(open(f'../data/wiki/wiki_{self.type}_dev.pkl', 'rb'))
            debias_test_raw = pickle.load(open(f'../data/wiki/wiki_{self.type}_test.pkl', 'rb'))
        else:
            debias_train = Path(f'../data/wiki/wiki_{self.type}_train.csv')
            debias_dev = Path(f'../data/wiki/wiki_{self.type}_dev.csv')
            debias_test = Path(f'../data/wiki/wiki_{self.type}_test.csv')

            # Optimize this later. We don't need pandas dataframe
            debias_train_raw = self.transform_dataframe_to_dict(data_frame=pd.read_csv(debias_train), tokenizer=self.tokenizer)
            debias_dev_raw = self.transform_dataframe_to_dict(data_frame=pd.read_csv(debias_dev), tokenizer=self.tokenizer)
            debias_test_raw = self.transform_dataframe_to_dict(data_frame=pd.read_csv(debias_test), tokenizer=self.tokenizer)


            pickle.dump(debias_train_raw, open(f'../data/wiki/wiki_{self.type}_train.pkl', 'wb'))
            pickle.dump(debias_dev_raw, open(f'../data/wiki/wiki_{self.type}_dev.pkl', 'wb'))
            pickle.dump(debias_test_raw, open(f'../data/wiki/wiki_{self.type}_test.pkl', 'wb'))

        # infer number of lables here
        number_of_labels = len(list(set([d['lable'] for d in debias_dev_raw])))
        if number_of_labels > 100:
            number_of_labels = 1 # if there are too many labels, most probably it is a regression task and not classifiation

        if self.vocab:
            vocab = self.vocab
        else:
            vocab = self.build_vocab_from_data(raw_train_data=debias_train_raw, raw_dev_data=debias_dev_raw,
                                      artificial_populate=self.artificial_populate)

        train_data = self.process_data(raw_data=debias_train_raw, vocab=vocab)
        dev_data = self.process_data(raw_data=debias_dev_raw, vocab=vocab)
        test_data = self.process_data(raw_data=debias_test_raw, vocab=vocab)

        self.pad_idx = vocab[self.pad_token]


        print("creating training data iterators")
        train_iterator = torch.utils.data.DataLoader(train_data,
                                                     self.batch_size,
                                                     shuffle=True,
                                                     collate_fn=self.collate)

        dev_iterator = torch.utils.data.DataLoader(dev_data,
                                                   self.batch_size,
                                                   shuffle=False,
                                                   collate_fn=self.collate)

        test_iterator = torch.utils.data.DataLoader(test_data,
                                                    self.batch_size,
                                                    shuffle=False,
                                                    collate_fn=self.collate)

        # number_of_labels = len(list(set(train_data.get_labels())))

        if number_of_labels > 100:
            number_of_labels = 1 # if there are too many labels, most probably it is a regression task and not classifiation

        return vocab, number_of_labels, train_iterator, dev_iterator, test_iterator, 0


class ValencePrediction(WikiSimpleClassification):

    def __init__(self, dataset_name:str,**params):
        super().__init__(dataset_name,**params)
        self.prepend_text = Template('Valence-reg-En-$split.txt')
        self.splits = ['train', 'dev', 'test-gold']
        self.data_dir = Path('../data/affect_in_tweet/task1/V-reg/')

    def transform_dataframe_to_dict(self, data, tokenizer):
        final_data = []
        for i in data:
            temp = {
                'text': clean_text_tweet(i[1]),
                'emotion': i[2],
                'lable': float(i[-1]),
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

    def read_data(self, data_dir, split):
        a = open(data_dir / Path(self.prepend_text.substitute(split=split)), 'r')
        lines = a.readlines()
        final_data = [l.strip().split('\t') for l in lines[1:]]
        return final_data

    def remove_data_without_label(self, data):
        new_data = []
        for d in data:
            if len(d[1]) > 5 and d[-1] != 'NONE' and float(d[-1]) > 0.001:
                new_data.append(d)
        return new_data

    def run(self):


        train = self.read_data(self.data_dir, 'train')

        dev = self.read_data(self.data_dir, 'dev')

        test = self.read_data(self.data_dir, 'test-gold')

        print(f"length of train is {len(train)}, length of dev is {len(dev)}, and length of test is {len(test)}")

        train = self.remove_data_without_label(train)
        dev = self.remove_data_without_label(dev)
        test = self.remove_data_without_label(test)

        print(
            f"After cleanup: length of train is {len(train)}, length of dev is {len(dev)}, and length of test is {len(test)}")

        train_processed = self.transform_dataframe_to_dict(data=train, tokenizer=self.tokenizer)
        dev_processed = self.transform_dataframe_to_dict(data=dev, tokenizer=self.tokenizer)
        test_processed = self.transform_dataframe_to_dict(data=test, tokenizer=self.tokenizer)

        # infer number of lables here
        number_of_labels = len(list(set([d['lable'] for d in dev_processed])))
        if number_of_labels > 100:
            number_of_labels = 1 # if there are too many labels, most probably it is a regression task and not classifiation

        if self.vocab:
            vocab = self.vocab
        else:
            vocab = self.build_vocab_from_data(raw_train_data=dev_processed, raw_dev_data=train_processed,
                                           artificial_populate=self.artificial_populate)


        train_data = self.process_data(raw_data=train_processed, vocab=vocab)
        dev_data = self.process_data(raw_data=dev_processed, vocab=vocab)
        test_data = self.process_data(raw_data=test_processed, vocab=vocab)

        self.pad_idx = vocab[self.pad_token]


        print("creating training data iterators")
        train_iterator = torch.utils.data.DataLoader(train_data,
                                                     self.batch_size,
                                                     shuffle=True,
                                                     collate_fn=self.collate)

        dev_iterator = torch.utils.data.DataLoader(dev_data,
                                                   self.batch_size,
                                                   shuffle=False,
                                                   collate_fn=self.collate)

        test_iterator = torch.utils.data.DataLoader(test_data,
                                                    self.batch_size,
                                                    shuffle=False,
                                                    collate_fn=self.collate)

        # number_of_labels = len(list(set(train_data.get_labels())))

        # if number_of_labels > 100:
        #     number_of_labels = 1 # if there are too many labels, most probably it is a regression task and not classifiation

        return vocab, number_of_labels, train_iterator, dev_iterator, test_iterator, 0

class BiasinBiosSimple(WikiSimpleClassification):

    def __init__(self, dataset_name: str, **params):
        super().__init__(dataset_name, **params)
        self.data_dir = Path('../data/bias_in_bios/') # don't really need it. The path is hard-coded

    def read_data(self, path):
        with open(path, "rb") as f:
            data = pickle.load(f)
        return data

    def transform_dataframe_to_dict(self, data, tokenizer, profession_to_id):
        """
        Although the data is not a dataframe. Keeping the same name so as to reflect legacy.
        :param data:
        :param tokenizer: this will be a simple splitting based on space.
        :return:
        """
        new_data = []
        for d in data:
            temp = {
                'lable': profession_to_id[d['p']],
                'original_text': d['text'],
                'text': d['hard_text'],
                'text_without_gender': d['text_without_gender'],
                'tokenized_text': tokenizer.tokenize(d['hard_text']) # this is a split by space tokenizer.
            }
            new_data.append(temp)

        return new_data


    def run(self):

        assert self.is_regression == False

        if self.trim_data:
            train = self.read_data("../data/bias_in_bios/train.pickle")[:15000]
            dev = self.read_data("../data/bias_in_bios/dev.pickle")[:500]
            test = self.read_data("../data/bias_in_bios/test.pickle")[:500]
        else:
            train = self.read_data("../data/bias_in_bios/train.pickle")
            dev = self.read_data("../data/bias_in_bios/dev.pickle")
            test = self.read_data("../data/bias_in_bios/test.pickle")


        # Find all professional. Create a professional to id list
        all_profession = list(set([t['p'] for t in train]))
        profession_to_id = {profession:index for index, profession in enumerate(all_profession)}
        pickle.dump(profession_to_id, open(self.data_dir / Path('profession_to_id.pickle'), "wb"))

        # Tokenization and id'fying the profession
        train_processed = self.transform_dataframe_to_dict(data=train, tokenizer=self.tokenizer, profession_to_id=profession_to_id)
        dev_processed = self.transform_dataframe_to_dict(data=dev, tokenizer=self.tokenizer, profession_to_id=profession_to_id)
        test_processed = self.transform_dataframe_to_dict(data=test, tokenizer=self.tokenizer, profession_to_id=profession_to_id)

        number_of_labels = len(all_profession)


        if self.vocab:
            vocab = self.vocab
        else:
            vocab = self.build_vocab_from_data(raw_train_data=dev_processed, raw_dev_data=train_processed,
                                           artificial_populate=self.artificial_populate)

        train_data = self.process_data(raw_data=train_processed, vocab=vocab)
        dev_data = self.process_data(raw_data=dev_processed, vocab=vocab)
        test_data = self.process_data(raw_data=test_processed, vocab=vocab)

        self.pad_idx = vocab[self.pad_token]


        print("creating training data iterators")
        train_iterator = torch.utils.data.DataLoader(train_data,
                                                     self.batch_size,
                                                     shuffle=True,
                                                     collate_fn=self.collate)

        dev_iterator = torch.utils.data.DataLoader(dev_data,
                                                   self.batch_size,
                                                   shuffle=False,
                                                   collate_fn=self.collate)

        test_iterator = torch.utils.data.DataLoader(test_data,
                                                    self.batch_size,
                                                    shuffle=False,
                                                    collate_fn=self.collate)

        # number_of_labels = len(list(set(train_data.get_labels())))

        if number_of_labels > 100:
            number_of_labels = 1 # if there are too many labels, most probably it is a regression task and not classifiation

        return vocab, number_of_labels, train_iterator, dev_iterator, test_iterator, 0

class BiasinBiosSimpleAdv(WikiSimpleClassification):

    def __init__(self, dataset_name: str, **params):
        super().__init__(dataset_name, **params)
        self.data_dir = Path('../data/bias_in_bios/') # don't really need it. The path is hard-coded
        self.sample_specific_class = params['sample_specific_class']
        self.classes_to_sample = ['physician', 'attorney']
        # List of all professions:
        '''
        ['architect', 'dietitian', 'surgeon', 'software_engineer', 'journalist', 'interior_designer', 
        'psychologist', 'composer', 'chiropractor', 'accountant', 'model', 'personal_trainer',
         'comedian', 'painter', 'paralegal', 'pastor', 'poet', 'filmmaker', 'rapper', 'attorney', 
         'teacher', 'photographer', 'dj', 'yoga_teacher', 'professor', 'dentist', 'nurse', 'physician']
         '''

    def read_data(self, path):
        with open(path, "rb") as f:
            data = pickle.load(f)
        return data

    def process_data(self, raw_data, vocab):
        """raw data is assumed to be tokenized"""
        final_data = [(data_point['lable'], data_point['tokenized_text'], data_point['gender']) for data_point in raw_data]
        text_transformation = sequential_transforms(vocab_func(vocab),
                                                    totensor(dtype=torch.long))
        if self.is_regression:
            label_transform = sequential_transforms(totensor(dtype=torch.float))
        else:
            label_transform = sequential_transforms(totensor(dtype=torch.long))

        gender_transformation = sequential_transforms(totensor(dtype=torch.long))

        transforms = (label_transform, text_transformation, gender_transformation) # Needs to be in the same order as final data elements

        return TextClassificationDataset(final_data, vocab, transforms)


    def transform_dataframe_to_dict(self, data, tokenizer, profession_to_id, gender_to_id):
        """
        Although the data is not a dataframe. Keeping the same name so as to reflect legacy.
        :param data:
        :param tokenizer: this will be a simple splitting based on space.
        :return:
        """
        new_data = []
        for d in data:
            temp = {
                'lable': profession_to_id[d['p']],
                'original_text': d['text'],
                'text': d['hard_text'],
                'text_without_gender': d['text_without_gender'],
                'tokenized_text': tokenizer.tokenize(d['hard_text']), # this is a split by space tokenizer.
                'gender': gender_to_id[d['g']]
            }
            new_data.append(temp)

        return new_data

    def collate(self, batch):
        labels, text, gender = zip(*batch)
        if self.is_regression:
            labels = torch.FloatTensor(labels)
        else:
            labels = torch.LongTensor(labels)
        lengths = torch.LongTensor([len(x) for x in text])
        text = nn.utils.rnn.pad_sequence(text, padding_value=self.pad_idx)
        gender = torch.LongTensor(gender)

        return labels, text, lengths, gender


    def run(self):



        assert self.is_regression == False

        if self.trim_data:
            train = self.read_data("../data/bias_in_bios/train.pickle")[:15000]
            dev = self.read_data("../data/bias_in_bios/dev.pickle")
            test = self.read_data("../data/bias_in_bios/test.pickle")
        else:
            train = self.read_data("../data/bias_in_bios/train.pickle")
            dev = self.read_data("../data/bias_in_bios/dev.pickle")
            test = self.read_data("../data/bias_in_bios/test.pickle")

        if self.sample_specific_class:
            train = [t for t in train if t['p'] in self.classes_to_sample]
            dev = [t for t in train if t['p'] in self.classes_to_sample]
            test = [t for t in train if t['p'] in self.classes_to_sample]

        if self.sample_specific_class:
            all_profession = list(set([t['p'] for t in train]))
            profession_to_id = {profession: index for index, profession in enumerate(all_profession)}
            # not pickling them as this might change over time.
        else:
            # Find all professional. Create a professional to id list.
            try:
                profession_to_id =  pickle.load(open(self.data_dir / Path('profession_to_id.pickle'), "rb"))
                all_profession = profession_to_id.keys()
            except:
                all_profession = list(set([t['p'] for t in train]))
                profession_to_id = {profession:index for index, profession in enumerate(all_profession)}
                pickle.dump(profession_to_id, open(self.data_dir / Path('profession_to_id.pickle'), "wb"))

        # Find all genders and assign them id
        try:
            gender_to_id = pickle.load(open(self.data_dir / Path('gender_to_id.pickle'), "rb"))
        except:
            all_gender = list(set([t['g'] for t in train]))
            gender_to_id = {profession:index for index, profession in enumerate(all_gender)}
            pickle.dump(gender_to_id, open(self.data_dir / Path('gender_to_id.pickle'), "wb"))


        # Tokenization and id'fying the profession
        train_processed = self.transform_dataframe_to_dict(data=train, tokenizer=self.tokenizer,
                                                           profession_to_id=profession_to_id, gender_to_id=gender_to_id)
        dev_processed = self.transform_dataframe_to_dict(data=dev, tokenizer=self.tokenizer,
                                                         profession_to_id=profession_to_id, gender_to_id=gender_to_id)
        test_processed = self.transform_dataframe_to_dict(data=test, tokenizer=self.tokenizer,
                                                          profession_to_id=profession_to_id, gender_to_id=gender_to_id)

        number_of_labels = len(all_profession)


        if self.vocab:
            vocab = self.vocab
        else:
            vocab = self.build_vocab_from_data(raw_train_data=dev_processed, raw_dev_data=train_processed,
                                           artificial_populate=self.artificial_populate)

        train_data = self.process_data(raw_data=train_processed, vocab=vocab)
        dev_data = self.process_data(raw_data=dev_processed, vocab=vocab)
        test_data = self.process_data(raw_data=test_processed, vocab=vocab)

        self.pad_idx = vocab[self.pad_token]


        print("creating training data iterators")
        train_iterator = torch.utils.data.DataLoader(train_data,
                                                     self.batch_size,
                                                     shuffle=False,
                                                     collate_fn=self.collate
                                                     )

        dev_iterator = torch.utils.data.DataLoader(dev_data,
                                                   512,
                                                   shuffle=False,
                                                   collate_fn=self.collate
                                                   )

        test_iterator = torch.utils.data.DataLoader(test_data,
                                                    512,
                                                    shuffle=False,
                                                    collate_fn=self.collate
                                                    )

        # number_of_labels = len(list(set(train_data.get_labels())))

        if number_of_labels > 100:
            number_of_labels = 1 # if there are too many labels, most probably it is a regression task and not classifiation

        return vocab, number_of_labels, train_iterator, dev_iterator, test_iterator, len(gender_to_id)


class SimpleAdvDatasetReader():
    def __init__(self, dataset_name:str,**params):
        self.dataset_name = dataset_name.lower()
        self.batch_size = params['batch_size']
        self.train_split = .80

        if 'celeb' in self.dataset_name:
            self.X, self.y, self.s = drh.get_celeb_data()
        elif 'adult' in self.dataset_name and 'multigroup' not in self.dataset_name:
            self.X, self.y, self.s = drh.get_adult_data()
        elif 'crime' in self.dataset_name:
            self.X, self.y, self.s = drh.get_crimeCommunities_data()
        elif 'dutch' in self.dataset_name:
            self.X, self.y, self.s = drh.get_dutch_data()
        elif 'compas' in self.dataset_name:
            self.X, self.y, self.s = drh.get_compas_data()
        elif 'german' in self.dataset_name:
            self.X, self.y, self.s = drh.get_german_data()
        elif 'adult' in self.dataset_name and 'multigroup' in self.dataset_name:
            self.X, self.y, self.s = drh.get_celeb_multigroups_data()
        elif 'gaussian' in self.dataset_name:
            raise NotImplementedError
            # self.X, self.y, self.s = drh.get_gaussian_data(50000)
        else:
            raise NotImplementedError

        # converting all -1,1 -> 0,1
        self.y = (self.y+1)/2

        if len(np.unique(self.s)) == 2 and -1 in np.unique(self.s):
            self.s = (self.s + 1) / 2


    def process_data(self, X,y,s, vocab):
        """raw data is assumed to be tokenized"""
        final_data = [(a,b,c) for a,b,c in zip(y,X,s)]


        label_transform = sequential_transforms()
        input_transform = sequential_transforms()
        aux_transform = sequential_transforms()

        transforms = (label_transform, input_transform, aux_transform)

        return TextClassificationDataset(final_data, vocab, transforms)


    def collate(self, batch):
        labels, input, aux = zip(*batch)

        labels = torch.LongTensor(labels)
        aux = torch.LongTensor(aux)
        lengths = torch.LongTensor([len(x) for x in input])
        input = torch.FloatTensor(input)

        return labels, input, lengths, aux


    def run(self):
        dataset_size = self.X.shape[0] # examples*feature_size
        test_index = int(self.train_split*dataset_size)
        dev_index = int(self.train_split*dataset_size) - int(self.train_split*dataset_size*.10)

        number_of_labels = len(np.unique(self.y))

        train_X, train_y, train_s = self.X[:dev_index,:], self.y[:dev_index], self.s[:dev_index]
        dev_X, dev_y, dev_s = self.X[dev_index:test_index, :], self.y[dev_index:test_index], self.s[dev_index:test_index]
        test_X, test_y, test_s = self.X[test_index:, :], self.y[test_index:], self.s[test_index:]

        vocab = {'<pad>':1} # no need of vocab in these dataset. It is there for code compatibility purposes.

        train_data = self.process_data(train_X,train_y,train_s, vocab=vocab)
        dev_data = self.process_data(dev_X,dev_y,dev_s, vocab=vocab)
        test_data = self.process_data(test_X,test_y,test_s, vocab=vocab)

        train_iterator = torch.utils.data.DataLoader(train_data,
                                                     self.batch_size,
                                                     shuffle=False,
                                                     collate_fn=self.collate
                                                     )

        dev_iterator = torch.utils.data.DataLoader(dev_data,
                                                   512,
                                                   shuffle=False,
                                                   collate_fn=self.collate
                                                   )

        test_iterator = torch.utils.data.DataLoader(test_data,
                                                    512,
                                                    shuffle=False,
                                                    collate_fn=self.collate
                                                    )

        return vocab, number_of_labels, train_iterator, dev_iterator, test_iterator, len(np.unique(self.s))



class EncodedEmoji:
    def __init__(self, dataset_name, **params):
        self.batch_size = params['batch_size']
        self.dataset_name = dataset_name
        self.n = 100000 # https://github.com/HanXudong/Diverse_Adversaries_for_Mitigating_Bias_in_Training/blob/b5b4c99ada17b3c19ab2ae8789bb56058cb72643/scripts_deepmoji.py#L270
        self.folder_location = '../data/deepmoji'
        try:
            self.ratio = params['ratio_of_pos_neg']
        except:
            self.ratio = 0.8 # this the default in https://arxiv.org/pdf/2101.10001.pdf
        self.batch_size = params['batch_size']

    def read_data_file(self, input_file: str):
        vecs = np.load(input_file)

        np.random.shuffle(vecs)

        return vecs[:40000], vecs[40000:42000], vecs[42000:44000]

    def process_data(self, X,y,s, vocab):
        """raw data is assumed to be tokenized"""
        final_data = [(a,b,c) for a,b,c in zip(y,X,s)]


        label_transform = sequential_transforms()
        input_transform = sequential_transforms()
        aux_transform = sequential_transforms()

        transforms = (label_transform, input_transform, aux_transform)

        return TextClassificationDataset(final_data, vocab, transforms)


    def collate(self, batch):
        labels, input, aux = zip(*batch)

        labels = torch.LongTensor(labels)
        aux = torch.LongTensor(aux)
        lengths = torch.LongTensor([len(x) for x in input])
        input = torch.FloatTensor(input)

        return labels, input, lengths, aux

    def run(self):

        try:

            train_pos_pos, dev_pos_pos, test_pos_pos = self.read_data_file(f"{self.folder_location}/pos_pos.npy")
            train_pos_neg, dev_pos_neg, test_pos_neg = self.read_data_file(f"{self.folder_location}/pos_neg.npy")
            train_neg_pos, dev_neg_pos, test_neg_pos = self.read_data_file(f"{self.folder_location}/neg_pos.npy")
            train_neg_neg, dev_neg_neg, test_neg_neg = self.read_data_file(f"{self.folder_location}/neg_neg.npy")

        except:

            self.folder_location = '/home/gmaheshwari/storage/fair_nlp_dataset/data/deepmoji'
            train_pos_pos, dev_pos_pos, test_pos_pos = self.read_data_file(f"{self.folder_location}/pos_pos.npy")
            train_pos_neg, dev_pos_neg, test_pos_neg = self.read_data_file(f"{self.folder_location}/pos_neg.npy")
            train_neg_pos, dev_neg_pos, test_neg_pos = self.read_data_file(f"{self.folder_location}/neg_pos.npy")
            train_neg_neg, dev_neg_neg, test_neg_neg = self.read_data_file(f"{self.folder_location}/neg_neg.npy")


        n_1 = int(self.n * self.ratio / 2)
        n_2 = int(self.n * (1 - self.ratio) / 2)

        fnames = ['pos_pos.npy', 'pos_neg.npy', 'neg_pos.npy', 'neg_neg.npy']
        main_labels = [1, 1, 0, 0]
        protected_labels = [1, 0, 1, 0]
        ratios = [n_1, n_2, n_2, n_1]
        data = [train_pos_pos, train_pos_neg, train_neg_pos, train_neg_neg]

        X_train, y_train, s_train = [], [], []

        # loading data for train

        for data_file, main_label, protected_label, ratio in zip(data, main_labels, protected_labels, ratios):
            X_train = X_train + list(data_file[:ratio])
            y_train = y_train + [main_label] * len(data_file[:ratio])
            s_train = s_train + [protected_label] * len(data_file[:ratio])


        X_dev, y_dev, s_dev = [], [], []
        for data_file, main_label, protected_label in zip([dev_pos_pos, dev_pos_neg, dev_neg_pos, dev_neg_neg]
                , main_labels, protected_labels):
            X_dev = X_dev + list(data_file)
            y_dev = y_dev + [main_label] * len(data_file)
            s_dev = s_dev + [protected_label] * len(data_file)


        X_test, y_test, s_test = [], [], []
        for data_file, main_label, protected_label in zip([test_pos_pos, test_pos_neg, test_neg_pos, test_neg_neg]
                , main_labels, protected_labels):
            X_test = X_test + list(data_file)
            y_test = y_test + [main_label] * len(data_file)
            s_test = s_test + [protected_label] * len(data_file)


        X_train, y_train, s_train = np.asarray(X_train), np.asarray(y_train), np.asarray(s_train)
        X_dev, y_dev, s_dev = np.asarray(X_dev), np.asarray(y_dev), np.asarray(s_dev)
        X_test, y_test, s_test = np.asarray(X_test), np.asarray(y_test), np.asarray(s_test)


        vocab = {'<pad>':1} # no need of vocab in these dataset. It is there for code compatibility purposes.
        number_of_labels = 2

        # shuffling data
        shuffle_train_index = np.random.permutation(len(X_train))
        X_train, y_train, s_train = X_train[shuffle_train_index], y_train[shuffle_train_index], s_train[shuffle_train_index]

        shuffle_dev_index = np.random.permutation(len(X_dev))
        X_dev, y_dev, s_dev = X_dev[shuffle_dev_index], y_dev[shuffle_dev_index], s_dev[
            shuffle_dev_index]

        shuffle_test_index = np.random.permutation(len(X_test))
        X_test, y_test, s_test = X_test[shuffle_test_index], y_test[shuffle_test_index], s_test[
            shuffle_test_index]

        train_data = self.process_data(X_train,y_train,s_train, vocab=vocab)
        dev_data = self.process_data(X_dev,y_dev,s_dev, vocab=vocab)
        test_data = self.process_data(X_test,y_test,s_test, vocab=vocab)


        train_iterator = torch.utils.data.DataLoader(train_data,
                                                     self.batch_size,
                                                     shuffle=False,
                                                     collate_fn=self.collate
                                                     )

        dev_iterator = torch.utils.data.DataLoader(dev_data,
                                                   512,
                                                   shuffle=False,
                                                   collate_fn=self.collate
                                                   )

        test_iterator = torch.utils.data.DataLoader(test_data,
                                                    512,
                                                    shuffle=False,
                                                    collate_fn=self.collate
                                                    )

        return vocab, number_of_labels, train_iterator, dev_iterator, test_iterator, 2


class EncodedDpNLP:
    """Implements a set of dataset used by Differentially Private Representation for NLP: Formal Guarantee and An Empirical Study on Privacy and Fairness"""
    def __init__(self, dataset_name, **params):
        self.batch_size = params['batch_size']
        self.dataset_name = dataset_name
        if self.dataset_name == 'blog':
            self.index_for_s = 0
        elif self.dataset_name == 'blog_v2':
            self.index_for_s = 1
        else:
            raise NotImplementedError
        if 'blog' in self.dataset_name: # blog dataset
            self.file_name = [] # @TODO: find a location and save it.
            self.file_name.append('../data/dpnlp/encoded_data/blog.pkl')
            self.file_name.append('/home/gmaheshwari/storage/dpnlp/encoded_data/blog.pkl')
        else:
            raise NotImplementedError

    def process_data(self, X,y,s, vocab):
        """raw data is assumed to be tokenized"""
        final_data = [(a,b,c) for a,b,c in zip(y,X,s)]


        label_transform = sequential_transforms()
        input_transform = sequential_transforms()
        aux_transform = sequential_transforms()

        transforms = (label_transform, input_transform, aux_transform)

        return TextClassificationDataset(final_data, vocab, transforms)


    def collate(self, batch):
        labels, input, aux = zip(*batch)

        labels = torch.LongTensor(labels)
        aux = torch.LongTensor(aux)
        lengths = torch.LongTensor([len(x) for x in input])
        input = torch.FloatTensor(input)

        return labels, input, lengths, aux

    def run(self):

        try:
            data = pickle.load(open(self.file_name[0], 'rb'))
        except FileNotFoundError:
            data = pickle.load(open(self.file_name[1], 'rb'))

        y_train = np.asarray([int(d.label) for d in data.get_train_examples()])
        s_train = np.asarray([int(d.aux_label[self.index_for_s]) for d in data.get_train_examples()])
        X_train = np.asarray(data.get_train_encoding())

        y_dev = np.asarray([int(d.label) for d in data.get_dev_examples()])
        s_dev = np.asarray([int(d.aux_label[self.index_for_s]) for d in data.get_dev_examples()])
        X_dev = np.asarray(data.get_dev_encoding())

        y_test = np.asarray([int(d.label) for d in data.get_test_examples()])
        s_test = np.asarray([int(d.aux_label[self.index_for_s]) for d in data.get_test_examples()])
        X_test = np.asarray(data.get_test_encoding())

        vocab = {'<pad>':1} # no need of vocab in these dataset. It is there for code compatibility purposes.
        number_of_labels = len(data.get_labels())


        # shuffling data
        shuffle_train_index = np.random.permutation(len(X_train))
        X_train, y_train, s_train = X_train[shuffle_train_index], y_train[shuffle_train_index], s_train[shuffle_train_index]

        shuffle_dev_index = np.random.permutation(len(X_dev))
        X_dev, y_dev, s_dev = X_dev[shuffle_dev_index], y_dev[shuffle_dev_index], s_dev[
            shuffle_dev_index]

        shuffle_test_index = np.random.permutation(len(X_test))
        X_test, y_test, s_test = X_test[shuffle_test_index], y_test[shuffle_test_index], s_test[
            shuffle_test_index]

        train_data = self.process_data(X_train,y_train,s_train, vocab=vocab)
        dev_data = self.process_data(X_dev,y_dev,s_dev, vocab=vocab)
        test_data = self.process_data(X_test,y_test,s_test, vocab=vocab)


        train_iterator = torch.utils.data.DataLoader(train_data,
                                                     self.batch_size,
                                                     shuffle=False,
                                                     collate_fn=self.collate
                                                     )

        dev_iterator = torch.utils.data.DataLoader(dev_data,
                                                   512,
                                                   shuffle=False,
                                                   collate_fn=self.collate
                                                   )

        test_iterator = torch.utils.data.DataLoader(test_data,
                                                    512,
                                                    shuffle=False,
                                                    collate_fn=self.collate
                                                    )

        return vocab, number_of_labels, train_iterator, dev_iterator, test_iterator, number_of_labels


class DomainAdaptationAmazon:
    """
    In a manner a more unique task when compared to the above ones. However, once the test/train/valid have
    been appropriately calibrated. THis would be straightforward.
    """

    def __init__(self, dataset_name, **params):
        self.batch_size = params['batch_size']
        self.dataset_name = dataset_name # dataset_name amazon_sourceDataset_targetDataset
        # source_name = 'dvd'; target_name = 'electronics'; dataset_name = amazon_dvd_electronics
        self.source_name, self.target_name = dataset_name.split('_')[1:]
        self.file_location = []
        self.file_location.append('../data/amazon/')
        self.file_location.append('/home/gmaheshwari/storage/amazon/')

    def process_data(self, X,y,s, vocab):
        """raw data is assumed to be tokenized"""
        final_data = [(a,b,c) for a,b,c in zip(y,X,s)]


        label_transform = sequential_transforms()
        input_transform = sequential_transforms()
        aux_transform = sequential_transforms()

        transforms = (label_transform, input_transform, aux_transform)

        return TextClassificationDataset(final_data, vocab, transforms)


    def collate(self, batch):
        labels, input, aux = zip(*batch)

        labels = torch.LongTensor(labels)
        aux = torch.LongTensor(aux)
        lengths = torch.LongTensor([len(x) for x in input])
        input = torch.FloatTensor(input)

        return labels, input, lengths, aux

    def load_amazon(self, source_name, target_name):
        try:
            location = self.file_location[0]
            source_file = location + source_name + '_train.svmlight'
            target_file = location + target_name + '_train.svmlight'
            test_file = location + target_name + '_test.svmlight'
            xs, ys, xt, yt, xt_test, yt_test = load_svmlight_files([source_file, target_file, test_file])
        except FileNotFoundError:
            location = self.file_location[1]
            print(f"in except {location}")
            source_file = location + source_name + '_train.svmlight'
            target_file = location + target_name + '_train.svmlight'
            test_file = location + target_name + '_test.svmlight'
            xs, ys, xt, yt, xt_test, yt_test = load_svmlight_files([source_file, target_file, test_file])

        ys, yt, yt_test = (np.array((y + 1) / 2, dtype=int) for y in (ys, yt, yt_test))

        return xs.A, ys, xt.A, yt, xt_test.A, yt_test # .A converts sparse matrix to dense matrix.

    def run(self):
        xs, ys, xt, yt, xt_test, yt_test = self.load_amazon(self.source_name, self.target_name)
        print("we are in the run!")

        # shuffling data
        shuffle_ind = np.random.permutation(len(xs))
        xs, ys, xt, yt = xs[shuffle_ind], ys[shuffle_ind], xt[shuffle_ind], yt[shuffle_ind]

        # split the train data into validation and train
        '''
        As per the paper: Wasserstein Distance Guided Representation Learning for Domain Adaptation.
        
        We follow Long et al. (Transfer Feature Learning with Joint Distribution Adaptation) and evaluate all compared 
        approaches through grid search on the hyperparameter space, and
        report the best results of each approach.
            -src https://github.com/RockySJ/WDGRL/issues/5
        
        This implies that the hyper-params were choosen based on the test set. Thus for now our validation
        set is same as test set.  
        '''
        #
        vocab = {'<pad>':1} # no need of vocab in these dataset. It is there for code compatibility purposes.
        train_src_data = self.process_data(xs, ys, np.zeros_like(ys), vocab=vocab) # src s=0
        train_target_data = self.process_data(xt, yt, np.ones_like(yt), vocab=vocab)   # target s=1
        test_data = self.process_data(xt_test, yt_test, np.ones_like(yt_test), vocab=vocab)

        train_src_iterator = torch.utils.data.DataLoader(train_src_data,
                                                     self.batch_size,
                                                     shuffle=False,
                                                     collate_fn=self.collate
                                                     )

        train_target_iterator = torch.utils.data.DataLoader(train_target_data,
                                                         self.batch_size,
                                                         shuffle=False,
                                                         collate_fn=self.collate
                                                         )

        train_iterator = CombinedIterator(train_src_iterator, train_target_iterator)

        test_iterator = torch.utils.data.DataLoader(test_data,
                                                    512,
                                                    shuffle=False,
                                                    collate_fn=self.collate
                                                    )

        # dev/valid is same as test as we are optimizing over the test set. See comments above.
        dev_iterator = torch.utils.data.DataLoader(test_data,
                                                    512,
                                                    shuffle=False,
                                                    collate_fn=self.collate
                                                    )

        number_of_labels = len(np.unique(yt))

        return vocab, number_of_labels, train_iterator, dev_iterator, test_iterator, number_of_labels