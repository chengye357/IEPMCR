import torch as th
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from pprg_end2end.enc2dec.base_modules import BaseRNN
import ipdb

class EncoderRNN(BaseRNN):
    def __init__(self, input_dropout_p, rnn_cell, input_size, hidden_size, num_layers, output_dropout_p, bidirectional, variable_lengths):
        super(EncoderRNN, self).__init__(input_dropout_p=input_dropout_p, 
                                         rnn_cell=rnn_cell, 
                                         input_size=input_size, 
                                         hidden_size=hidden_size, 
                                         num_layers=num_layers, 
                                         output_dropout_p=output_dropout_p, 
                                         bidirectional=bidirectional)
        self.variable_lengths = variable_lengths
        self.output_size = hidden_size*2 if bidirectional else hidden_size

    def forward(self, input_var, init_state=None, input_lengths=None, goals=None):
        embedded = self.input_dropout(input_var)
        if init_state is not None:
            output, hidden = self.rnn(embedded, init_state)
        else:
            output, hidden = self.rnn(embedded)
        return output, hidden


class CtxEncoder(nn.Module):
    def __init__(self, vocab_size, embedding_dim, rnn_cell,
                 hidden_size, num_layers, input_dropout_p, output_dropout_p,
                 bidirectional, variable_lengths, use_attn, embedding=None):
        super(CtxEncoder, self).__init__()
        if embedding is None:
            self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embedding_dim)
        else:
            self.embedding = embedding

        self.rnn = EncoderRNN(input_dropout_p=input_dropout_p,
                              rnn_cell=rnn_cell, 
                              input_size=embedding_dim,
                              hidden_size=hidden_size,
                              num_layers=num_layers, 
                              output_dropout_p=output_dropout_p, 
                              bidirectional=bidirectional, 
                              variable_lengths=variable_lengths)

        self.hidden_size = hidden_size
        self.multiplier = 2 if bidirectional else 1
        self.output_size = self.multiplier * self.hidden_size
        self.use_attn = use_attn
        if self.use_attn:
            self.key_w = nn.Linear(self.output_size, self.hidden_size)#600 300
            self.query = nn.Linear(self.hidden_size, 1)#300 1

#     def forward(self, utterances, init_state=None):
    def forward(self, ctx_feature, ctx_mask, init_state=None):
#         ipdb.set_trace()
#         batch_size, max_ctx_len = utterances.size()
#         flat_words = utterances.view(-1, max_ctx_len) # (batch_size, max_ctx_len)
#         word_embeddings = self.embedding(flat_words) # (batch_size, max_ctx_len, embedding_dim)
#         flat_mask = th.sign(flat_words).float()#sign函数，[1,1,1,0,0,0,0]区分id与pad

        batch_size, max_ctx_len = ctx_mask.size()
        word_embeddings = ctx_feature
#         flat_mask = th.sign(ctx_id).float()
        flat_mask = ctx_mask
    
        # enc_outs: (batch_size, max_ctx_len, num_directions*hidden_size)
        # enc_last: (num_layers*num_directions, batch_size, hidden_size)
        enc_outs, enc_last = self.rnn(word_embeddings, init_state=init_state)

        if self.use_attn:
#             ipdb.set_trace()
            fc1 = th.tanh(self.key_w(enc_outs)) # (batch_size, max_ctx_len, hidden_size)
            attn = self.query(fc1).squeeze(2)  # (batch_size, max_ctx_len)
            attn = F.softmax(attn, attn.dim()-1) # (batch_size, max_ctx_len, 1)
            attn = attn * flat_mask
            attn = (attn / (th.sum(attn, dim=1, keepdim=True)+1e-10)).unsqueeze(2)
            ctx_embedded = attn * enc_outs # (batch_size, max_ctx_len, num_directions*hidden_size)
            ctx_embedded = th.sum(ctx_embedded, dim=1) # (batch_size, num_directions*hidden_size)
        else:
            attn = None
            ctx_embedded = enc_last.transpose(0, 1).contiguous() # (batch_size, num_layers*num_directions, hidden_size)
            ctx_embedded = ctx_embedded.view(-1, self.output_size) # (batch_size*num_layers, num_directions*hidden_size)

        ctx_embedded = ctx_embedded.view(batch_size, self.output_size)#32 600
        # ctx_embedded = ctx_embedded[:, :self.hidden_size] + ctx_embedded[:, self.hidden_size:]

        return ctx_embedded, \
               enc_outs.contiguous().view(batch_size, max_ctx_len, -1), \
               flat_mask==0
        # bs, 2*hidden_size; bs, max_rsp_len, 2*hidden_size; bs, max_rsp_len


class RspEncoder(nn.Module):
    def __init__(self, vocab_size, embedding_dim, rnn_cell,
                 hidden_size, num_layers, input_dropout_p, output_dropout_p,
                 bidirectional, variable_lengths, use_attn, embedding=None):
        super(RspEncoder, self).__init__()
        if embedding is None:
            self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embedding_dim)
        else:
            self.embedding = embedding

        self.rnn = EncoderRNN(input_dropout_p=input_dropout_p,
                              rnn_cell=rnn_cell,
                              input_size=embedding_dim,
                              hidden_size=hidden_size,
                              num_layers=num_layers,
                              output_dropout_p=output_dropout_p,
                              bidirectional=bidirectional,
                              variable_lengths=variable_lengths)

        self.hidden_size = hidden_size
        self.multiplier = 2 if bidirectional else 1
        self.output_size = self.multiplier * self.hidden_size
        self.use_attn = use_attn
        if self.use_attn:
            self.key_w = nn.Linear(self.output_size, self.hidden_size)
            self.query = nn.Linear(self.hidden_size, 1)
            
    def forward(self, rsp_feature, rsp_mask, init_state=None):

        batch_size, max_rsp_len = rsp_mask.size()
        word_embeddings = rsp_feature
        flat_mask = rsp_mask
        enc_outs, enc_last = self.rnn(word_embeddings, init_state=init_state)

#     def forward(self, utterances, init_state=None):
#         batch_size, max_rsp_len = utterances.size()
#         flat_words = utterances.view(-1, max_rsp_len) # (batch_size, max_rsp_len)
#         word_embeddings = self.embedding(flat_words) # (batch_size, max_rsp_len, embedding_dim)
#         flat_mask = th.sign(flat_words).float()

#         # enc_outs: (batch_size, max_rsp_len, num_directions*hidden_size)
#         # enc_last: (num_layers*num_directions, batch_size, hidden_size)
#         enc_outs, enc_last = self.rnn(word_embeddings, init_state=init_state)

        if self.use_attn:
            fc1 = th.tanh(self.key_w(enc_outs)) # (batch_size, max_rsp_len, hidden_size)
            # print ('This is the size of fc1: {}.'.format(fc1.size()))
            attn = self.query(fc1).squeeze(2)  # (batch_size, max_rsp_len)
            # print ('This is the size of attn: {}.'.format(attn.size()))
            attn = F.softmax(attn, attn.dim()-1) # (batch_size, max_rsp_len, 1)
            attn = attn * flat_mask
            attn = (attn / (th.sum(attn, dim=1, keepdim=True)+1e-10)).unsqueeze(2)
            # print ('This is the unsqueezed attn size: {}.'.format(attn.size()))
            rsp_embedded = attn * enc_outs # (batch_size, max_rsp_len, num_directions*hidden_size)
            rsp_embedded = th.sum(rsp_embedded, dim=1) # (batch_size, num_directions*hidden_size)
        else:
            attn = None
            rsp_embedded = enc_last.transpose(0, 1).contiguous() # (batch_size, num_layers*num_directions, hidden_size)
            rsp_embedded = rsp_embedded.view(-1, self.output_size) # (batch_size*num_layers, num_directions*hidden_size)

        rsp_embedded = rsp_embedded.view(batch_size, self.output_size)
        # rsp_embedded = rsp_embedded[:, :self.hidden_size] + rsp_embedded[:, self.hidden_size:]
        return rsp_embedded, \
               enc_outs.contiguous().view(batch_size, max_rsp_len, -1), \
               flat_mask==0
        # bs, 2*hidden_size; bs, max_rsp_len, 2*hidden_size; bs, max_rsp_len
        
class CtxEncoder_2(nn.Module):
    def __init__(self, vocab_size, embedding_dim, rnn_cell,
                 hidden_size, num_layers, input_dropout_p, output_dropout_p,
                 bidirectional, variable_lengths, use_attn, embedding=None):
        super(CtxEncoder_2, self).__init__()
        if embedding is None:
            self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embedding_dim)
        else:
            self.embedding = embedding

        self.rnn = EncoderRNN(input_dropout_p=input_dropout_p,
                              rnn_cell=rnn_cell, 
                              input_size=embedding_dim,
                              hidden_size=hidden_size,
                              num_layers=num_layers, 
                              output_dropout_p=output_dropout_p, 
                              bidirectional=bidirectional, 
                              variable_lengths=variable_lengths)

        self.hidden_size = hidden_size
        self.multiplier = 2 if bidirectional else 1
        self.output_size = self.multiplier * self.hidden_size
        self.use_attn = use_attn
        if self.use_attn:
            self.key_w = nn.Linear(self.output_size, self.hidden_size)#600 300
            self.query = nn.Linear(self.hidden_size, 1)#300 1

    def forward(self, utterances, init_state=None):
#     def forward(self, ctx_feature, ctx_mask, init_state=None):
#         ipdb.set_trace()
        batch_size, max_ctx_len = utterances.size()
        flat_words = utterances.view(-1, max_ctx_len) # (batch_size, max_ctx_len)
        word_embeddings = self.embedding(flat_words) # (batch_size, max_ctx_len, embedding_dim)
        flat_mask = th.sign(flat_words).float()#sign函数，[1,1,1,0,0,0,0]区分id与pad
    
        # enc_outs: (batch_size, max_ctx_len, num_directions*hidden_size)
        # enc_last: (num_layers*num_directions, batch_size, hidden_size)
        enc_outs, enc_last = self.rnn(word_embeddings, init_state=init_state)

        if self.use_attn:
#             ipdb.set_trace()
            fc1 = th.tanh(self.key_w(enc_outs)) # (batch_size, max_ctx_len, hidden_size)
            attn = self.query(fc1).squeeze(2)  # (batch_size, max_ctx_len)
            attn = F.softmax(attn, attn.dim()-1) # (batch_size, max_ctx_len, 1)
            attn = attn * flat_mask
            attn = (attn / (th.sum(attn, dim=1, keepdim=True)+1e-10)).unsqueeze(2)
            ctx_embedded = attn * enc_outs # (batch_size, max_ctx_len, num_directions*hidden_size)
            ctx_embedded = th.sum(ctx_embedded, dim=1) # (batch_size, num_directions*hidden_size)
        else:
            attn = None
            ctx_embedded = enc_last.transpose(0, 1).contiguous() # (batch_size, num_layers*num_directions, hidden_size)
            ctx_embedded = ctx_embedded.view(-1, self.output_size) # (batch_size*num_layers, num_directions*hidden_size)

        ctx_embedded = ctx_embedded.view(batch_size, self.output_size)#32 600
        # ctx_embedded = ctx_embedded[:, :self.hidden_size] + ctx_embedded[:, self.hidden_size:]

        return ctx_embedded, \
               enc_outs.contiguous().view(batch_size, max_ctx_len, -1), \
               flat_mask==0
        # bs, 2*hidden_size; bs, max_rsp_len, 2*hidden_size; bs, max_rsp_len


class RspEncoder_2(nn.Module):
    def __init__(self, vocab_size, embedding_dim, rnn_cell,
                 hidden_size, num_layers, input_dropout_p, output_dropout_p,
                 bidirectional, variable_lengths, use_attn, embedding=None):
        super(RspEncoder_2, self).__init__()
        if embedding is None:
            self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embedding_dim)
        else:
            self.embedding = embedding

        self.rnn = EncoderRNN(input_dropout_p=input_dropout_p,
                              rnn_cell=rnn_cell,
                              input_size=embedding_dim,
                              hidden_size=hidden_size,
                              num_layers=num_layers,
                              output_dropout_p=output_dropout_p,
                              bidirectional=bidirectional,
                              variable_lengths=variable_lengths)

        self.hidden_size = hidden_size
        self.multiplier = 2 if bidirectional else 1
        self.output_size = self.multiplier * self.hidden_size
        self.use_attn = use_attn
        if self.use_attn:
            self.key_w = nn.Linear(self.output_size, self.hidden_size)
            self.query = nn.Linear(self.hidden_size, 1)
            

    def forward(self, utterances, init_state=None):
        batch_size, max_rsp_len = utterances.size()
        flat_words = utterances.view(-1, max_rsp_len) # (batch_size, max_rsp_len)
        word_embeddings = self.embedding(flat_words) # (batch_size, max_rsp_len, embedding_dim)
        flat_mask = th.sign(flat_words).float()

        # enc_outs: (batch_size, max_rsp_len, num_directions*hidden_size)
        # enc_last: (num_layers*num_directions, batch_size, hidden_size)
        enc_outs, enc_last = self.rnn(word_embeddings, init_state=init_state)

        if self.use_attn:
            fc1 = th.tanh(self.key_w(enc_outs)) # (batch_size, max_rsp_len, hidden_size)
            # print ('This is the size of fc1: {}.'.format(fc1.size()))
            attn = self.query(fc1).squeeze(2)  # (batch_size, max_rsp_len)
            # print ('This is the size of attn: {}.'.format(attn.size()))
            attn = F.softmax(attn, attn.dim()-1) # (batch_size, max_rsp_len, 1)
            attn = attn * flat_mask
            attn = (attn / (th.sum(attn, dim=1, keepdim=True)+1e-10)).unsqueeze(2)
            # print ('This is the unsqueezed attn size: {}.'.format(attn.size()))
            rsp_embedded = attn * enc_outs # (batch_size, max_rsp_len, num_directions*hidden_size)
            rsp_embedded = th.sum(rsp_embedded, dim=1) # (batch_size, num_directions*hidden_size)
        else:
            attn = None
            rsp_embedded = enc_last.transpose(0, 1).contiguous() # (batch_size, num_layers*num_directions, hidden_size)
            rsp_embedded = rsp_embedded.view(-1, self.output_size) # (batch_size*num_layers, num_directions*hidden_size)

        rsp_embedded = rsp_embedded.view(batch_size, self.output_size)
        # rsp_embedded = rsp_embedded[:, :self.hidden_size] + rsp_embedded[:, self.hidden_size:]
        return rsp_embedded, \
               enc_outs.contiguous().view(batch_size, max_rsp_len, -1), \
               flat_mask==0
        # bs, 2*hidden_size; bs, max_rsp_len, 2*hidden_size; bs, max_rsp_len