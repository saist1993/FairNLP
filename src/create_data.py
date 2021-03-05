# reads, cleans, tokenizes data and create data specific iterators, and vocabulary

import torch
import pickle
import torchtext
import collections
import pandas as pd
import torch.nn as nn
from pathlib import Path
from string import Template
from typing import List, Callable


from utils import totensor, sequential_transforms, vocab_func, TextClassificationDataset

class WikiSimpleClassification:

    def __init__(self, dataset_name:str,**params):
        '''
        reads, cleans, tokenizes data and create data specific iterators, and vocabulary for wiki (toxicity) tasks.
        More info here:
        '''
        self.dataset_name = dataset_name
        self.tokenizer = params['tokenizer']
        self.type = dataset_name.split('_')[1]  # type can be debias, main etc.
        self.artificial_populate = params['artificial_populate']
        self.pad_token = params['pad_token']
        self.batch_size = params['batch_size']
        self.is_regression = params['is_regression']
        self.pad_idx = -1 # this gets updated in the run function once the vocab is made.

    def transform_dataframe_to_dict(self, data_frame:pd, tokenizer:Callable):
        """

        reads the actual data from the disk
        @ still need to make it more general, specifcally path needs to be read from some external file.
        """
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

        number_of_labels = len(list(set(train_data.get_labels())))

        if number_of_labels > 100:
            number_of_labels = 1 # if there are too many labels, most probably it is a regression task and not classifiation

        return vocab, number_of_labels, train_iterator, dev_iterator, test_iterator



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
                'text': i[1],
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

        number_of_labels = len(list(set(train_data.get_labels())))

        if number_of_labels > 100:
            number_of_labels = 1 # if there are too many labels, most probably it is a regression task and not classifiation

        return vocab, number_of_labels, train_iterator, dev_iterator, test_iterator