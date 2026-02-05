import torch as th
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from pprg_end2end.enc2dec.base_modules import BaseRNN
from pprg_end2end.enc2dec.decoders import Attention


class RspDecoder_simpler(nn.Module):
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

        # Get this step's decoder hidden state
        outut, hidden = self.gru(dec_input, prev_h)   # [B ,1, H], [1, B, H]

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

    def forward_simrsp(self, hidden, sim_rsp_enc_hidden, sim_rsp_enc_pad_mask, coverage):
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

        # Get this step's decoder hidden state
        # outut, hidden = self.gru(dec_input, prev_h)   # [B ,1, H], [1, B, H]

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
        return  attn_dist, sim_rsp_vec