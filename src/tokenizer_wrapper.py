# a set of class which wraps different tokenizer.

from nltk.tokenize import TweetTokenizer
from tqdm.auto import tqdm
import spacy

# custom imports
from utils import clean_text

class SpacyTokenizer:
    """Cleans the data and tokenizes it"""

    def __init__(self, spacy_model: str = "en_core_web_sm", clean_text=clean_text, max_length=None):
        self.tokenizer_model = spacy.load(spacy_model)
        self.clean_text = clean_text
        self.max_length = max_length

    def tokenize(self, s):
        if self.clean_text:
            s = self.clean_text(s)
        doc = self.tokenizer_model(s)
        tokens = [token.text for token in doc]

        if self.max_length:
            tokens = tokens[:self.max_length]

        return tokens

    def batch_tokenize(self, texts: list):
        """tokenizes a list via nlp pipeline space"""
        nlp = self.tokenizer_model

        tokenized_list = []

        if self.max_length:
            for doc in tqdm(nlp.pipe(texts, disable=["ner", "tok2vec"])):
                tokenized_list.append([t.text for t in doc][:self.max_length])
        else:
            for doc in tqdm(nlp.pipe(texts, disable=["ner", "tok2vec"])):
                tokenized_list.append([t.text for t in doc])

        return tokenized_list


class TwitterTokenizer:
    """Cleans the data and tokenizes it"""

    def __init__(self, clean_text=clean_text, max_length=None):
        self.tokenizer_model = TweetTokenizer()
        self.clean_text = clean_text
        self.max_length = max_length

    def tokenize(self, s):

        tokens = self.tokenizer_model.tokenize(s)
        final_token = []
        if self.clean_text:
            for t in tokens:
                clean_t = self.clean_text(t)
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