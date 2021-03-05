import torch
import torch.nn as nn

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

        # hidden = [batch size, hid dim * 2]

        prediction = self.fc(self.dropout(hidden))

        # prediction = [batch size, output dim]

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