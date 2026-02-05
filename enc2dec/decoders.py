import torch as th
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from pprg_end2end.enc2dec.base_modules import BaseRNN
import ipdb

class Attention(nn.Module):
    """
    Attention mechanism based on Bahdanau et al. (2015) - Eq. (1)(2)
    augmented with Coverage mechanism - Eq. (11)
    B : batch size
    L : source text length
    H : encoder hidden state dimension
    """

    def __init__(self, hidden_dim, use_coverage):
        super().__init__()
        # Eq. (1)
        self.v = nn.Linear(hidden_dim * 2, 1, bias=False)                       # v
        self.enc_proj = nn.Linear(hidden_dim * 2, hidden_dim * 2, bias=False)   # W_h
        self.dec_proj = nn.Linear(hidden_dim, hidden_dim * 2, bias=True)        # W_s, b_attn

        self.use_coverage = use_coverage
        if self.use_coverage:
            # Additional parameter for coverage vector; w_c in Eq. (11)
            self.w_c = nn.Linear(1, hidden_dim * 2, bias=False)

    def forward(self, dec_input, coverage, enc_hidden, enc_pad_mask):
        """
        Args:
            dec_input: decoder hidden state             [1, B, H]
            coverage: coverage vector                   [B x L]
            enc_hidden: encoder hidden states           [B x L x 2H]
            enc_pad_mask: encoder padding masks         [B x L]

        Returns:
            attn_dist: attention dist'n over src tokens [B x L]
        """
#         ipdb.set_trace()
        # Eq. (1)
        dec_input = dec_input.squeeze(0)  # b , h
        enc_feature = self.enc_proj(enc_hidden)         # [B x L x 2H]
        dec_feature = self.dec_proj(dec_input)          # [B x 2H]
        dec_feature = dec_feature.unsqueeze(1)          # [B x 1 x 2H]
        scores = enc_feature + dec_feature              # [B x L x 2H]

        if self.use_coverage:
            # Eq. (11)
            coverage = coverage.unsqueeze(-1)           # [B x L x 1]
            cov_feature = self.w_c(coverage)            # [B x L x 2H]
            scores = scores + cov_feature

        scores = th.tanh(scores)                     # [B x L x 2H]
        scores = self.v(scores)                         # [B x L x 1]
        scores = scores.squeeze(-1)                     # [B x L]

        # Don't attend over padding; fill '-inf' where enc_pad_mask == True
        if enc_pad_mask is not None:
            scores = scores.float().masked_fill_(
                enc_pad_mask,
                float('-inf')
            ).type_as(scores)  # FP16 support: cast to float and back

        # Eq. (2)
        attn_dist = F.softmax(scores, dim=-1)               # [B x L]

        return attn_dist


class RspDecoder(nn.Module):
    """
    Single-layer unidirectional GRU with attention for a single timestep - Eq. (3)(4)
    B : batch size
    E : embedding size
    H : decoder hidden state dimension
    V : vocab size
    """

    def __init__(self, input_dim, hidden_dim, vocab_size, use_coverage):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.gru = nn.GRU(input_size=input_dim, hidden_size=hidden_dim, batch_first=True)

        self.attention_withctx = Attention(hidden_dim, use_coverage)
        # self.v_withctx = nn.Linear(hidden_dim * 3, hidden_dim, bias=True)   # V, b
        # self.v_out_withctx = nn.Linear(hidden_dim, vocab_size, bias=True)   # V', b'

        self.attention_withrsp = Attention(hidden_dim, use_coverage)
        # self.v_withrsp = nn.Linear(hidden_dim * 3, hidden_dim, bias=True)  # V, b
        # self.v_out_withrsp = nn.Linear(hidden_dim, vocab_size, bias=True)  # V', b'

    def forward(self, dec_input, prev_h, ctx_enc_hidden, ctx_enc_pad_mask, coverage):
        """
        Args:
            dec_input: decoder input embedding at timestep t    [B x 1 x E]
            prev_h: decoder hidden state from prev timestep     [1 x B x H]
            ctx_enc_hidden: encoder hidden states                   [B x L x 2H]
            ctx_enc_pad_mask: encoder masks for attn computation    [B x L]
            sim_rsp_enc_hidden: encoder hidden states                   [B x L' x 2H]
            sim_rsp_enc_pad_mask: encoder masks for attn computation    [B x L']
            coverage: coverage vector at timestep t - Eq. (10)  [B x L]

        Returns:
            vocab_dist: predicted vocab dist'n at timestep t    [B x V]
            ctx_attn_dist: attention dist'n at timestep t           [B x L]
            sim_rsp_attn_dist: attention dist'n at timestep t           [B x L']
            context_vec: context vector at timestep t           [B x H]
            hidden: hidden state at timestep t                  [B x H]
            cell: cell state at timestep t                      [B x H]
        """
#         ipdb.set_trace()
        # Get this step's decoder hidden state
        output, hidden = self.gru(dec_input, prev_h)   # [B ,1, H], [1, B, H]

        # Compute attention distribution over enc states
        attn_dist = self.attention_withctx(dec_input=hidden,
                                   coverage=coverage,
                                   enc_hidden=ctx_enc_hidden,
                                   enc_pad_mask=ctx_enc_pad_mask)   # [B x L]

        # Eq. (3) - Sum weighted enc hidden states to make context vector
        # The context vector is used later to compute generation probability
        context_vec = th.bmm(attn_dist.unsqueeze(1), ctx_enc_hidden)     # [B x 1 x H]
        context_vec = th.sum(context_vec, dim=1)                     # [B x 2H]

        # # Eq. (4)
        # output = self.v_withctx(th.cat([hidden, context_vec], dim=-1))       # [B x 3H] -> [B x H]
        # output = self.v_out_withctx(output)                                     # [B x V]
        # vocab_dist = F.softmax(output, dim=-1)                          # [B x V]
        # return vocab_dist, attn_dist, context_vec, hidden
        return attn_dist, context_vec, hidden

    def forward_simrsp(self, dec_input, prev_h, sim_rsp_enc_hidden, sim_rsp_enc_pad_mask, coverage):
        """
        Args:
            dec_input: decoder input embedding at timestep t    [B x 1 x E]
            prev_h: decoder hidden state from prev timestep     [1 x B x H]
            ctx_enc_hidden: encoder hidden states                   [B x L x 2H]
            ctx_enc_pad_mask: encoder masks for attn computation    [B x L]
            sim_rsp_enc_hidden: encoder hidden states                   [B x L' x 2H]
            sim_rsp_enc_pad_mask: encoder masks for attn computation    [B x L']
            coverage: coverage vector at timestep t - Eq. (10)  [B x L]

        Returns:
            vocab_dist: predicted vocab dist'n at timestep t    [B x V]
            ctx_attn_dist: attention dist'n at timestep t           [B x L]
            sim_rsp_attn_dist: attention dist'n at timestep t           [B x L']
            context_vec: context vector at timestep t           [B x H]
            hidden: hidden state at timestep t                  [B x H]
            cell: cell state at timestep t                      [B x H]
        """
#         ipdb.set_trace()
        # Get this step's decoder hidden state
        outut, hidden = self.gru(dec_input, prev_h)   # [B ,1, H], [1, B, H]

        # Compute attention distribution over enc states
        attn_dist = self.attention_withrsp(dec_input=hidden,
                                   coverage=coverage,
                                   enc_hidden=sim_rsp_enc_hidden,
                                   enc_pad_mask=sim_rsp_enc_pad_mask)   # [B x L]

        # Eq. (3) - Sum weighted enc hidden states to make context vector
        # The context vector is used later to compute generation probability
        sim_rsp_vec = th.bmm(attn_dist.unsqueeze(1), sim_rsp_enc_hidden)     # [B x 1 x H]
        sim_rsp_vec = th.sum(sim_rsp_vec, dim=1)                     # [B x 2H]

        # # Eq. (4)
        # output = self.v_withrsp(th.cat([hidden, sim_rsp_vec], dim=-1))       # [B x 3H] -> [B x H]
        # output = self.v_out_withrsp(output)                                     # [B x V]
        # vocab_dist = F.softmax(output, dim=-1)                          # [B x V]
        return  attn_dist, sim_rsp_vec, hidden