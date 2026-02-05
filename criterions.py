import torch as th
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.loss import _Loss

class Loss(nn.Module):
    """
    Computes nll loss (Eq. (6)), coverage loss (Eq. (12)),
    and the composite loss function that combines the two (Eq. (13)).
    """
    def __init__(self, pad_id, cov_weight, use_coverage):
        super().__init__()
        self.use_coverage = use_coverage
        self.cov_weight = cov_weight   # hyperparameter lambda in Eq. (13)
        self.pad_id = pad_id

    def nll_loss(self, output, target):
        """
        Negative log likelihood of target word - Eq. (6)
        Args:
            output: predicted probs from each timestep      [B x V_x T]
            target: answer ids using extended vocab         [B x T]

        Returns:
            loss: nll loss value; averaged over batch & timestep
        """
        output = th.log(output)
        loss = F.nll_loss(output, target,
                          ignore_index=self.pad_id,
                          reduction='mean')
        return loss

    def cov_loss(self, attn_dist, coverage, dec_pad_mask, dec_len):
        """
        Coverage loss at timestep t - Eq. (12)
        Args:
            attn_dist: attention distribution from all timesteps            [B x L x T]
            coverage: sum of previous attn dist's from all timesteps        [B x L x T]
            dec_pad_mask: target sequence padding masks [PAD] -> True       [B x T]
            dec_len: target sequence lengths                                [B]

        Returns:
            loss: coverage loss value; averaged over batch & timestep
        """
        min_val = th.min(attn_dist, coverage)    # [B x L x T]
        loss = th.sum(min_val, dim=1)            # [B x T]

        # ignore loss from [PAD] tokens
        loss = loss.masked_fill_(
            dec_pad_mask,
            0.0
        )
        avg_loss = th.sum(loss) / th.sum(dec_len)
        return avg_loss

    def forward(self, output, batch):
        """
        Eq. (13) - Composite loss
        Args:
            output: a dictionary of model outputs with the following keys
                - final_dist
                - attn_dist
                - coverage
            batch: `Batch` instance

        Returns:
            loss: final composite loss value
        """
        final_dist = output['final_dist']
        dec_target = batch.dec_target
        nll_loss = self.nll_loss(output=final_dist, target=dec_target)

        attn_dist = output['attn_dist']
        coverage = output['coverage']
        dec_pad_mask = batch.dec_pad_mask
        dec_len = batch.dec_len
        cov_loss = self.cov_loss(attn_dist, coverage, dec_pad_mask, dec_len)
        return nll_loss, cov_loss


class NLLEntropy(_Loss):
    def __init__(self, padding_idx, ):
        super(NLLEntropy, self).__init__()
        self.padding_idx = padding_idx

    def forward(self, net_output, labels):
        batch_size = net_output.size(0)
        pred = net_output.view(-1, net_output.size(-1))
        target = labels.view(-1)
        pred = th.log(pred)
        loss = F.nll_loss(pred, target, size_average=True, ignore_index=self.padding_idx)
        return loss


class NLLEntropy4CLF(_Loss):
    def __init__(self, dictionary, bad_tokens=['<disconnect>', '<disagree>'], reduction='elementwise_mean'):
        super(NLLEntropy4CLF, self).__init__()
        w = th.Tensor(len(dictionary)).fill_(1)
        for token in bad_tokens:
            w[dictionary[token]] = 0.0
        self.crit = nn.CrossEntropyLoss(w, reduction=reduction)

    def forward(self, preds, labels):
        # preds: (batch_size, outcome_len, outcome_vocab_size)
        # labels: (batch_size, outcome_len)
        preds = preds.view(-1, preds.size(-1))
        labels = labels.view(-1)
        return self.crit(preds, labels)



class CatKLLoss(_Loss):
    def __init__(self):
        super(CatKLLoss, self).__init__()

    def forward(self, log_qy, log_py, batch_size=None, unit_average=False):
        """
        qy * log(q(y)/p(y))
        """
        qy = th.exp(log_qy)
        y_kl = th.sum(qy * (log_qy - log_py), dim=1)
        if unit_average:
            return th.mean(y_kl)
        else:
            return th.sum(y_kl)/batch_size


class Entropy(_Loss):
    def __init__(self):
        super(Entropy, self).__init__()

    def forward(self, log_qy, batch_size=None, unit_average=False):
        """
        -qy log(qy)
        """
        if log_qy.dim() > 2:
            log_qy = log_qy.squeeze()
        qy = th.exp(log_qy)
        h_q = th.sum(-1 * log_qy * qy, dim=1)
        if unit_average:
            return th.mean(h_q)
        else:
            return th.sum(h_q) / batch_size


class BinaryNLLEntropy(_Loss):

    def __init__(self, size_average=True):
        super(BinaryNLLEntropy, self).__init__()
        self.size_average = size_average

    def forward(self, net_output, label_output):
        """
        :param net_output: batch_size x
        :param labels:
        :return:
        """
        batch_size = net_output.size(0)
        loss = F.binary_cross_entropy_with_logits(net_output, label_output, size_average=self.size_average)
        if self.size_average is False:
            loss /= batch_size
        return loss


class NormKLLoss(_Loss):
    def __init__(self, unit_average=False):
        super(NormKLLoss, self).__init__()
        self.unit_average = unit_average

    def forward(self, recog_mu, recog_logvar, prior_mu, prior_logvar):
        # find the KL divergence between two Gaussian distribution
        loss = 1.0 + (recog_logvar - prior_logvar)
        loss -= th.div(th.pow(prior_mu - recog_mu, 2), th.exp(prior_logvar))
        loss -= th.div(th.exp(recog_logvar), th.exp(prior_logvar))
        if self.unit_average:
            kl_loss = -0.5 * th.mean(loss, dim=1)
        else:
            kl_loss = -0.5 * th.sum(loss, dim=1)
        avg_kl_loss = th.mean(kl_loss)
        return avg_kl_loss
