import torch
import torch.nn as nn
import torch.nn.functional as func

from utils import GradReverse, laplace


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

        self.embedding = nn.Embedding(input_dim, emb_dim, padding_idx=pad_idx)
        self.lstm = nn.LSTM(emb_dim, hid_dim, num_layers=n_layers, bidirectional=True, dropout=dropout)
        self.fc = nn.Linear(2 * hid_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

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

        prediction = self.fc(self.dropout(hidden))

        # prediction = [batch size, output dim]

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
                x = func.relu(self.dropout(layer(x)))
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
            hidden = hidden + m.sample(hidden.shape).squeeze().to(self.device)



        prediction = self.fc(self.dropout(hidden))
        adv_output = self.adv(GradReverse.apply(hidden))

        # prediction = [batch size, output dim]
        #

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