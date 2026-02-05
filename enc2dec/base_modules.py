import torch as th
import torch.nn as nn
import numpy as np
from torch.nn.modules.module import _addindent


class BaseRNN(nn.Module):
    KEY_ATTN_SCORE = 'attention_score'
    KEY_SEQUENCE = 'sequence'

    def __init__(self, input_dropout_p, rnn_cell, 
                     input_size, hidden_size, num_layers, 
                     output_dropout_p, bidirectional):
        super(BaseRNN, self).__init__()
        self.input_dropout = nn.Dropout(p=input_dropout_p)
        if rnn_cell.lower() == 'lstm':
            self.rnn_cell = nn.LSTM
        elif rnn_cell.lower() == 'gru':
            self.rnn_cell = nn.GRU
        else:
            raise ValueError('Unsupported RNN Cell Type: {0}'.format(rnn_cell))
        self.rnn = self.rnn_cell(input_size=input_size, 
                                 hidden_size=hidden_size,
                                 num_layers=num_layers, 
                                 batch_first=True, 
                                 dropout=output_dropout_p, 
                                 bidirectional=bidirectional)

        if rnn_cell.lower() == 'lstm':
            for names in self.rnn._all_weights:
                for name in filter(lambda n: 'bias' in n, names):
                    bias = getattr(self.rnn, name)
                    n = bias.size(0)
                    start, end = n // 4, n // 2
                    bias.data[start:end].fill_(1.)
