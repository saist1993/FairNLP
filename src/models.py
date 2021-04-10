import torch
import torch.nn as nn
import torch.nn.functional as func

from utils import GradReverse, laplace


class CNN(nn.Module):
    def __init__(self, model_params):
        super().__init__()
        input_dim = model_params['input_dim']
        emb_dim = model_params['emb_dim']
        pad_idx = model_params['pad_idx']
        num_filters = model_params['num_filters']
        filter_sizes = model_params['filter_sizes']
        output_dim = model_params['output_dim']
        dropout = model_params['dropout']

        self.embedding = nn.Embedding(input_dim, emb_dim, padding_idx=pad_idx)
        self.convs = nn.ModuleList(
            [nn.Conv2d(
                in_channels=1,
                out_channels= num_filters,
                kernel_size= (fs, emb_dim)
            ) for fs in filter_sizes]
        )

        self.fc = nn.Linear(len(filter_sizes)*num_filters, output_dim)
        self.dropout = nn.Dropout(dropout)


    def forward(self, text, lengths):
        """Note that length is not used. Can be a dummy. Kept for consistency purpose"""
        embedded = self.embedding(text.transpose(1,0)) # text = bs*sl -> embedded = bs*sl*emb_dim
        embedded = embedded.unsqueeze(1) # bs*sl*emb_dim

        conved = [func.relu(conv(embedded)).squeeze(3) for conv in self.convs]

        # conved_n = [batch size, n_filters, sent len - filter_sizes[n] + 1]

        pooled = [func.max_pool1d(conv, conv.shape[2]).squeeze(2) for conv in conved]

        # pooled_n = [batch size, n_filters]

        cat = self.dropout(torch.cat(pooled, dim=1))

        # cat = [batch size, n_filters * len(filter_sizes)]

        return self.fc(cat)


class CNNAdv(nn.Module):
    def __init__(self, model_params):
        super().__init__()
        input_dim = model_params['input_dim']
        emb_dim = model_params['emb_dim']
        pad_idx = model_params['pad_idx']
        num_filters = model_params['num_filters']
        filter_sizes = model_params['filter_sizes']
        output_dim = model_params['output_dim']
        dropout = model_params['dropout']
        adv_number_of_layers = model_params['adv_number_of_layers']
        adv_dropout = model_params['adv_dropout']
        self.device = model_params['device']
        self.noise_layer = model_params['noise_layer']
        self.eps = model_params['eps']
        try:
            self.return_hidden = model_params['return_hidden']
        except KeyError:
            self.return_hidden = False


        self.embedding = nn.Embedding(input_dim, emb_dim, padding_idx=pad_idx)
        self.convs = nn.ModuleList(
            [nn.Conv2d(
                in_channels=1,
                out_channels= num_filters,
                kernel_size= (fs, emb_dim)
            ) for fs in filter_sizes]
        )

        self.fc = nn.Linear(len(filter_sizes)*num_filters, output_dim)
        self.dropout = nn.Dropout(dropout)

        self.adv = DomainAdv(number_of_layers=adv_number_of_layers, input_dim=len(filter_sizes)*num_filters,
                             hidden_dim=hid_dim, output_dim=2, dropout=adv_dropout)

        self.adv.apply(initialize_parameters) # don't know, if this is needed.


class BiLSTM(nn.Module):
    def __init__(self, model_params):
        super().__init__()

        input_dim = model_params['input_dim']
        emb_dim = model_params['emb_dim']
        hid_dim = model_params['hidden_dim']
        output_dim = model_params['output_dim']
        n_layers = model_params['n_layers']
        dropout = model_params['dropout']
        pad_idx = model_params['pad_idx']
        self.device = model_params['device']
        try:
            self.noise_layer = model_params['noise_layer']
        except KeyError:
            self.noise_layer = False

        try:
            self.eps = model_params['eps']
        except KeyError:
            self.eps = False

        self.embedding = nn.Embedding(input_dim, emb_dim, padding_idx=pad_idx)
        self.lstm = nn.LSTM(emb_dim, hid_dim, num_layers=n_layers, bidirectional=True, dropout=dropout)
        self.fc = nn.Linear(2 * hid_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, text, lengths, return_hidden=False):
        # text = [seq len, batch size]
        # lengths = [batch size]

        embedded = self.dropout(self.embedding(text))

        # embedded = [seq len, batch size, emb dim]

        packed_embedded = nn.utils.rnn.pack_padded_sequence(embedded, lengths, enforce_sorted=False)

        packed_output, (hidden, cell) = self.lstm(packed_embedded)

        output, _ = nn.utils.rnn.pad_packed_sequence(packed_output)

        # outputs = [seq_len, batch size, n directions * hid dim]
        # hidden = [n layers * n directions, batch size, hid dim]

        hidden_fwd = hidden[-2]
        hidden_bck = hidden[-1]

        # hidden_fwd/bck = [batch size, hid dim]

        original_hidden = torch.cat((hidden_fwd, hidden_bck), dim=1)

        # insert domain adversarial stuff here.

        # hidden = [batch size, hid dim * 2]



        # prediction = [batch size, output dim]
        if self.noise_layer:
            m = torch.distributions.laplace.Laplace(torch.tensor([0.0]), torch.tensor([laplace(self.eps, 2)]))
            # max_hidden = torch.max(hidden, 1, keepdims=True)[0]
            # min_hidden = torch.min(hidden, 1, keepdims=True)[0]
            # hidden = (hidden - min_hidden)/ (max_hidden - min_hidden)
            hidden = original_hidden / torch.norm(original_hidden, keepdim=True)
            hidden = hidden + m.sample(hidden.shape).squeeze().to(self.device)
        else:
            hidden = original_hidden

        prediction = self.fc(self.dropout(hidden))

        if return_hidden:
            return prediction, original_hidden, hidden

        return prediction

class DomainAdv(nn.Module):

    def __init__(self, number_of_layers:int, input_dim, hidden_dim, output_dim, dropout):
        super().__init__()
        self.fc_layers = []
        self.dropout = nn.Dropout(dropout)
        for i in range(number_of_layers):
            if i != number_of_layers - 1 and i != 0:
                self.fc_layers.append((nn.Linear(hidden_dim, hidden_dim)))
            elif i == 0:
                self.fc_layers.append(nn.Linear(input_dim, hidden_dim))
            else:
                self.fc_layers.append(nn.Linear(hidden_dim, output_dim)) # @TODO: see if there is a need for a softmax via sigmoid or something

        self.fc_layers = torch.nn.ModuleList(self.fc_layers)

    def forward(self, x):
        for index, layer in enumerate(self.fc_layers):
            if len(self.fc_layers)-1 != index:
                # x = func.relu(self.dropout(layer(x)))
                x = func.relu(layer(x))
            else:
                x = layer(x)
        return x



class BiLSTMAdv(nn.Module):
    def __init__(self, model_params):
        super().__init__()

        input_dim = model_params['input_dim']
        emb_dim = model_params['emb_dim']
        hid_dim = model_params['hidden_dim']
        output_dim = model_params['output_dim']
        n_layers = model_params['n_layers']
        dropout = model_params['dropout']
        pad_idx = model_params['pad_idx']
        adv_number_of_layers = model_params['adv_number_of_layers']
        adv_dropout = model_params['adv_dropout']
        self.device = model_params['device']
        self.noise_layer = model_params['noise_layer']
        self.eps = model_params['eps']
        try:
            self.return_hidden = model_params['return_hidden']
        except KeyError:
            self.return_hidden = False

        self.embedding = nn.Embedding(input_dim, emb_dim, padding_idx=pad_idx)
        self.lstm = nn.LSTM(emb_dim, hid_dim, num_layers=n_layers, bidirectional=True, dropout=dropout)
        self.fc = nn.Linear(2 * hid_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

        self.adv = DomainAdv(number_of_layers=adv_number_of_layers, input_dim=2*hid_dim,
                             hidden_dim=hid_dim, output_dim=2, dropout=adv_dropout)

        self.adv.apply(initialize_parameters) # don't know, if this is needed.


    def forward(self, text, lengths):
        # text = [seq len, batch size]
        # lengths = [batch size]

        embedded = self.dropout(self.embedding(text))

        # embedded = [seq len, batch size, emb dim]

        packed_embedded = nn.utils.rnn.pack_padded_sequence(embedded, lengths, enforce_sorted=False)

        packed_output, (hidden, cell) = self.lstm(packed_embedded)

        output, _ = nn.utils.rnn.pad_packed_sequence(packed_output)

        # outputs = [seq_len, batch size, n directions * hid dim]
        # hidden = [n layers * n directions, batch size, hid dim]

        hidden_fwd = hidden[-2]
        hidden_bck = hidden[-1]

        # hidden_fwd/bck = [batch size, hid dim]

        hidden = torch.cat((hidden_fwd, hidden_bck), dim=1)



        # insert domain adversarial stuff here.

        # hidden = [batch size, hid dim * 2]



        if self.noise_layer:
            m = torch.distributions.laplace.Laplace(torch.tensor([0.0]), torch.tensor([laplace(self.eps, 1)]))
            max_hidden = torch.max(hidden, 1, keepdims=True)[0]
            min_hidden = torch.min(hidden, 1, keepdims=True)[0]
            hidden = (hidden - min_hidden)/ (max_hidden - min_hidden)
            # hidden = hidden/torch.norm(hidden, keepdim=True)
            hidden = hidden + m.sample(hidden.shape).squeeze().to(self.device)


        # hidden = hidden/torch.norm(hidden, keepdim=True)

        prediction = self.fc(hidden)
        adv_output = self.adv(GradReverse.apply(hidden))
        # adv_output = prediction

        if self.return_hidden:
            return prediction, adv_output, hidden

        return prediction, adv_output


class BOWClassifier(nn.Module):
    def __init__(self, model_params):
        super().__init__()
        input_dim = model_params['input_dim']
        emb_dim = model_params['emb_dim']
        output_dim = model_params['output_dim']
        pad_idx = model_params['pad_idx']
        dropout = model_params['dropout']
        hidden_dim = model_params['hidden_dim']
        self.device = model_params['device']



        self.embedding = nn.Embedding(input_dim, emb_dim, padding_idx=pad_idx)
        self.fc1 = nn.Linear(emb_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, text, lengths):

        embedded = self.embedding(text)
        # a = [t[:length,:] for t, length in zip(embedded.transpose(1,0),lengths)]

        embedded = torch.stack([torch.mean(t[:length, :],0) for t, length in zip(embedded.transpose(1, 0), lengths)])
        prediction = torch.relu(self.dropout(self.fc1(embedded)))
        prediction = self.fc2(prediction)
        return prediction



class Attacker(nn.Module):
    def __init__(self, model_params, original_model):
        super(Attacker, self).__init__()
        self.original_model = original_model # model forward. This is the palce through which one will get hidden

        for param_group in self.original_model.parameters():
            param_group.requires_grad = False

        hid_dim = model_params['hidden_dim']
        adv_number_of_layers = model_params['adv_number_of_layers']
        adv_dropout = model_params['adv_dropout']
        self.device = model_params['device']
        self.adv = DomainAdv(number_of_layers=adv_number_of_layers, input_dim= 2*hid_dim,
                             hidden_dim=hid_dim, output_dim=2, dropout=adv_dropout)
    def forward(self, text, lengths):
        _, _, _,hidden = self.original_model(text, lengths, return_hidden=True)
        output = self.adv(hidden)
        return output


class EmbedderLSTM(nn.Module):
    def __init__(self, model_params):
        super().__init__()
        input_dim = model_params['input_dim']
        emb_dim = model_params['emb_dim']
        hid_dim = model_params['hidden_dim']
        output_dim = model_params['output_dim']
        n_layers = model_params['n_layers']
        dropout = model_params['dropout']
        pad_idx = model_params['pad_idx']

        self.embedding = nn.Embedding(input_dim, emb_dim, padding_idx=pad_idx)
        self.lstm = nn.LSTM(emb_dim, hid_dim, num_layers=n_layers, bidirectional=True, dropout=dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, text, lengths):
        # text = [seq len, batch size]
        # lengths = [batch size]

        # embedded = self.dropout(self.embedding(text))
        embedded = self.embedding(text)

        # embedded = [seq len, batch size, emb dim]

        packed_embedded = nn.utils.rnn.pack_padded_sequence(embedded, lengths, enforce_sorted=False)

        packed_output, (hidden, cell) = self.lstm(packed_embedded)

        output, _ = nn.utils.rnn.pad_packed_sequence(packed_output)

        # outputs = [seq_len, batch size, n directions * hid dim]
        # hidden = [n layers * n directions, batch size, hid dim]

        hidden_fwd = hidden[-2]
        hidden_bck = hidden[-1]

        # hidden_fwd/bck = [batch size, hid dim]

        hidden = torch.cat((hidden_fwd, hidden_bck), dim=1)

        return hidden


class BiLSTMAdvWithFreeze(nn.Module):
    def __init__(self, model_params):
        super().__init__()

        input_dim = model_params['input_dim']
        emb_dim = model_params['emb_dim']
        hid_dim = model_params['hidden_dim']
        output_dim = model_params['output_dim']
        n_layers = model_params['n_layers']
        dropout = model_params['dropout']
        pad_idx = model_params['pad_idx']
        adv_number_of_layers = model_params['adv_number_of_layers']
        adv_dropout = model_params['adv_dropout']
        self.device = model_params['device']
        self.noise_layer = model_params['noise_layer']
        self.eps = model_params['eps']
        self.learnable_embeddings = model_params['learnable_embeddings']

        self.legend = {
            'embedding': 0,
            'encoder': 1,
            'classifier': 2,
            'adversary': 3
        }

        try:
            self.return_hidden = model_params['return_hidden']
        except KeyError:
            self.return_hidden = False

        self.embedder = EmbedderLSTM(model_params)
        self.adv = DomainAdv(number_of_layers=1, input_dim=2*hid_dim,
                             hidden_dim=hid_dim, output_dim=2, dropout=adv_dropout)
        self.classifier = DomainAdv(number_of_layers=2, input_dim=2 * hid_dim,
                             hidden_dim=hid_dim, output_dim=output_dim, dropout=adv_dropout)
        self.adv.apply(initialize_parameters) # don't know, if this is needed.
        self.classifier.apply(initialize_parameters)  # don't know, if this is needed.
        self.embedder.apply(initialize_parameters)  # don't know, if this is needed.

    def freeze_unfreeze_adv(self, freeze=True):
        if freeze:
            for param_group in self.adv.parameters():
                param_group.requires_grad = False
        else:
            for param_group in self.adv.parameters():
                param_group.requires_grad = True

    def freeze_unfreeze_classifier(self, freeze=True):
        if freeze:
            for param_group in self.classifier.parameters():
                param_group.requires_grad = False
        else:
            for param_group in self.classifier.parameters():
                param_group.requires_grad = True

    def freeze_unfreeze_embedder(self, freeze=True):
        if freeze:
            for param_group in self.embedder.parameters():
                param_group.requires_grad = False
        else:
            for param_group in self.embedder.parameters():
                param_group.requires_grad = True
            if not self.learnable_embeddings:
                self.embedder.embedding.weight.requires_grad = False

    @property
    def layers(self):
        return torch.nn.ModuleList([self.embedder.embedding, torch.nn.ModuleList([self.embedder.lstm, self.embedder.dropout]),
                                   self.classifier, self.adv])

    def forward(self, text, lengths, gradient_reversal=False, return_hidden=False):
        # text = [seq len, batch size]
        # lengths = [batch size]



        original_hidden = self.embedder(text, lengths)


        if self.noise_layer:
            m = torch.distributions.laplace.Laplace(torch.tensor([0.0]), torch.tensor([laplace(self.eps, 2)]))
            # max_hidden = torch.max(hidden, 1, keepdims=True)[0]
            # min_hidden = torch.min(hidden, 1, keepdims=True)[0]
            # hidden = (hidden - min_hidden)/ (max_hidden - min_hidden)
            hidden = original_hidden/torch.norm(original_hidden, keepdim=True)
            hidden = hidden + m.sample(hidden.shape).squeeze().to(self.device)
        else:
            hidden = original_hidden


        # hidden = hidden/torch.norm(hidden, keepdim=True)

        prediction = self.classifier(hidden)
        if gradient_reversal:
            adv_output = self.adv(GradReverse.apply(hidden))
        else:
            adv_output = self.adv(hidden)

        if return_hidden:
            return prediction, adv_output, original_hidden, hidden

        return prediction, adv_output


def initialize_parameters(m):
    if isinstance(m, nn.Embedding):
        nn.init.uniform_(m.weight, -0.05, 0.05)
    elif isinstance(m, nn.LSTM):
        for n, p in m.named_parameters():
            if 'weight_ih' in n:
                i, f, g, o = p.chunk(4)
                nn.init.xavier_uniform_(i)
                nn.init.xavier_uniform_(f)
                nn.init.xavier_uniform_(g)
                nn.init.xavier_uniform_(o)
            elif 'weight_hh' in n:
                i, f, g, o = p.chunk(4)
                nn.init.orthogonal_(i)
                nn.init.orthogonal_(f)
                nn.init.orthogonal_(g)
                nn.init.orthogonal_(o)
            elif 'bias' in n:
                i, f, g, o = p.chunk(4)
                nn.init.zeros_(i)
                nn.init.ones_(f)
                nn.init.zeros_(g)
                nn.init.zeros_(o)
    elif isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        nn.init.zeros_(m.bias)