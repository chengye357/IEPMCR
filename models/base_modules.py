import os
import torch as th
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from transformers import BertTokenizer
from pprg_end2end.utils import INT, FLOAT, LONG, cast_type, summary
# from pprg_end2end.models import VisionTransformer
import ipdb
import logging
logger = logging.getLogger()

# def init_tokenizer():
#     tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
#     tokenizer.add_special_tokens({'bos_token':'[DEC]'})
#     tokenizer.add_special_tokens({'additional_special_tokens':['[ENC]']})       
#     tokenizer.enc_token_id = tokenizer.additional_special_tokens_ids[0]  
#     return tokenizer

# def create_vit(image_size, use_grad_checkpointing=False, ckpt_layer=0, drop_path_rate=0):
#     vision_width = 768
#     visual_encoder = VisionTransformer(
#         img_size=image_size, 
#         patch_size=16, 
#         embed_dim=vision_width, 
#         depth=12, 
#         num_heads=12, 
#         use_grad_checkpointing=use_grad_checkpointing,
#         ckpt_layer=ckpt_layer,
#         drop_path_rate=0 or drop_path_rate) 
#     return visual_encoder, vision_width

# def tie_encoder_decoder_weights(encoder: nn.Module, decoder: nn.Module, base_model_prefix: str, skip_key:str):
#     uninitialized_encoder_weights: List[str] = []
#     if decoder.__class__ != encoder.__class__:
#         logger.info(
#             f"{decoder.__class__} and {encoder.__class__} are not equal. In this case make sure that all encoder weights are correctly initialized."
#         )

#     def tie_encoder_to_decoder_recursively(
#         decoder_pointer: nn.Module,
#         encoder_pointer: nn.Module,
#         module_name: str,
#         uninitialized_encoder_weights: List[str],
#         skip_key: str,
#         depth=0,
#     ):
#         assert isinstance(decoder_pointer, nn.Module) and isinstance(
#             encoder_pointer, nn.Module
#         ), f"{decoder_pointer} and {encoder_pointer} have to be of type torch.nn.Module"
#         if hasattr(decoder_pointer, "weight") and skip_key not in module_name:
#             assert hasattr(encoder_pointer, "weight")
#             encoder_pointer.weight = decoder_pointer.weight
#             if hasattr(decoder_pointer, "bias"):
#                 assert hasattr(encoder_pointer, "bias")
#                 encoder_pointer.bias = decoder_pointer.bias                
#             print(module_name+' is tied')    
#             return

#         encoder_modules = encoder_pointer._modules
#         decoder_modules = decoder_pointer._modules
#         if len(decoder_modules) > 0:
#             assert (
#                 len(encoder_modules) > 0
#             ), f"Encoder module {encoder_pointer} does not match decoder module {decoder_pointer}"

#             all_encoder_weights = set([module_name + "/" + sub_name for sub_name in encoder_modules.keys()])
#             encoder_layer_pos = 0
#             for name, module in decoder_modules.items():
#                 if name.isdigit():
#                     encoder_name = str(int(name) + encoder_layer_pos)
#                     decoder_name = name
#                     if not isinstance(decoder_modules[decoder_name], type(encoder_modules[encoder_name])) and len(
#                         encoder_modules
#                     ) != len(decoder_modules):
#                         # this can happen if the name corresponds to the position in a list module list of layers
#                         # in this case the decoder has added a cross-attention that the encoder does not have
#                         # thus skip this step and subtract one layer pos from encoder
#                         encoder_layer_pos -= 1
#                         continue
#                 elif name not in encoder_modules:
#                     continue
#                 elif depth > 500:
#                     raise ValueError(
#                         "Max depth of recursive function `tie_encoder_to_decoder` reached. It seems that there is a circular dependency between two or more `nn.Modules` of your model."
#                     )
#                 else:
#                     decoder_name = encoder_name = name
#                 tie_encoder_to_decoder_recursively(
#                     decoder_modules[decoder_name],
#                     encoder_modules[encoder_name],
#                     module_name + "/" + name,
#                     uninitialized_encoder_weights,
#                     skip_key,
#                     depth=depth + 1,
#                 )
#                 all_encoder_weights.remove(module_name + "/" + encoder_name)

#             uninitialized_encoder_weights += list(all_encoder_weights)

#     # tie weights recursively
#     tie_encoder_to_decoder_recursively(decoder, encoder, base_model_prefix, uninitialized_encoder_weights, skip_key)  

class BaseModel(nn.Module):
    def __init__(self, config):
        super(BaseModel, self).__init__()
        self.use_gpu = config.use_gpu
        self.config = config
        self.kl_w = 0.0

    def np2var(self, inputs, dtype):
        if inputs is None:
            return None
        return cast_type(Variable(th.from_numpy(inputs)), 
                         dtype, 
                         self.use_gpu)
    def tensor2cuda(self, inputs, dtype):
        if inputs is None:
            return None
        return cast_type(inputs, dtype, self.use_gpu)
    
    def forward(self, *inputs):
        raise NotImplementedError

    def backward(self, loss, batch_cnt, add_loss_adv=None):
        total_loss = self.valid_loss(loss, batch_cnt, add_loss_adv=add_loss_adv)
        total_loss.backward()

    def valid_loss(self, loss, batch_cnt=None, add_loss_adv=None):
        total_loss = 0.0
        for k, l in loss.items():
            if l is not None:
                total_loss += l
        return total_loss

    def get_optimizer(self, config, verbose=True):
        if config.op == 'adam':
            if verbose:
                print('Use Adam')
#             op = optim.Adam([
#                 {'params':self.module_retrieval.parameters(), 'lr':config.init_lr_retrieval},
#                 {'params':self.module_reuse.parameters()}
            
#             ], lr=config.init_lr, weight_decay=config.l2_norm)
            
#             ipdb.set_trace()
            op = optim.Adam(filter(lambda p: p.requires_grad, self.parameters()), lr=config.init_lr,
                              weight_decay=config.l2_norm)
#             return optim.Adam(filter(lambda p: p.requires_grad, self.parameters()), lr=config.init_lr,
#                               weight_decay=config.l2_norm)
            return op
        elif config.op == 'sgd':
            print('Use SGD')
            return optim.SGD(self.parameters(), lr=config.init_lr, momentum=config.momentum)
        elif config.op == 'rmsprop':
            print('Use RMSProp')
            return optim.RMSprop(self.parameters(), lr=config.init_lr, momentum=config.momentum)

    def get_clf_optimizer(self, config):  # no usage
        params = []
        params.extend(self.gru_attn_encoder.parameters())
        params.extend(self.feat_projecter.parameters())
        params.extend(self.sel_classifier.parameters())

        if config.fine_tune_op == 'adam':
            print('Use Adam')
            return optim.Adam(params, lr=config.fine_tune_lr)
        elif config.fine_tune_op == 'sgd':
            print('Use SGD')
            return optim.SGD(params, lr=config.fine_tune_lr, momentum=config.fine_tune_momentum)
        elif config.fine_tune_op == 'rmsprop':
            print('Use RMSProp')
            return optim.RMSprop(params, lr=config.fine_tune_lr, momentum=config.fine_tune_momentum)

        
    def model_sel_loss(self, loss, batch_cnt):
        return self.valid_loss(loss, batch_cnt)


    def extract_short_ctx(self, context, context_lens, backward_size=1):
        utts = []
        # context = (batch_size, max_ctx_len, max_utt_len), context_len = (batch_size, ), number of turns
        if self.config.context_lens == 'long':
            for b_id in range(context.shape[0]):
                utts.append(np.concatenate(context[b_id]))  # a concatenation of all prev turns
                # (batch_size, max_ctx_len*max_utt_len)
        else:
            for b_id in range(context.shape[0]):
                # print ('This is the ctx len: ', context_lens[b_id])
                utts.append(context[b_id, context_lens[b_id]-1]) # only one prev turn
                # (batch_size, max_utt_len)
        return np.array(utts)


    def extract_short_in_act(self, in_acts, context_lens, backward_size=1):
        acts = []
        # in_acts = (batch_size, max_ctx_len, max_act_len), context_len = (batch_size, ), number of turns
        if self.config.context_lens == 'long':
            for b_id in range(in_acts.shape[0]):
                acts.append(np.concatenate(in_acts[b_id]))  # a concatenation of all prev turns
                # (batch_size, max_ctx_len*max_act_len)
        else:
            for b_id in range(in_acts.shape[0]):
                acts.append(in_acts[b_id, context_lens[b_id]-1]) # only one prev turn's usr action
                # (batch_size, max_act_len)
        return np.array(acts)

    def flatten_context(self, context, context_lens, align_right=False):  # no usage
        utts = []
        temp_lens = []
        for b_id in range(context.shape[0]):
            temp = []
            for t_id in range(context_lens[b_id]):
                for token in context[b_id, t_id]:
                    if token != 0:
                        temp.append(token)
            temp_lens.append(len(temp))
            utts.append(temp)
        max_temp_len = np.max(temp_lens)
        results = np.zeros((context.shape[0], max_temp_len))
        for b_id in range(context.shape[0]):
            if align_right:
                results[b_id, -temp_lens[b_id]:] = utts[b_id]
            else:
                results[b_id, 0:temp_lens[b_id]] = utts[b_id]

        return results

    def print_summary(self):
        logger.info(summary(self, show_weights=False))

    def load(self, path, model_id):
        """
            load {model_id}-model from {path}
        """
        self.load_state_dict(th.load(os.path.join(path, '{}-model'.format(model_id))))

    def save(self, path, model_id):
        """
            save {model_id}-model in {path}
        """
        th.save(self.state_dict(), os.path.join(path, '{}-model'.format(model_id)))
        
     
        
class MLPLayer(nn.Module):
    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.dense = nn.Linear(dim_in, dim_in)
        self.activation = nn.Tanh() # nn.ReLU()
        self.fc = nn.Linear(dim_in, dim_out)

    def forward(self, features, **kwargs):
        x = self.dense(features)
        x = self.activation(x)
        x = self.fc(x)
        # x = self.fc(features)
        return x

class TripletLoss(nn.Module):
    def __init__(self, margin=0.3):
        super(TripletLoss, self).__init__()
        self.margin = margin
        self.ranking_loss = nn.MarginRankingLoss(margin=margin)

    def forward(self, inputs, labels):
#         ipdb.set_trace()
        n = inputs.size(0)
        batch_size = n // 2
        # Compute pairwise distance, replace by the official when merged
        dist = th.pow(inputs, 2).sum(dim=1, keepdim=True).expand(n, n)#平方，channel加和，复制16列
        dist = dist + dist.t()  # dis[i][j] = a^2 + b^2
        dist.addmm_(1, -2, inputs, inputs.t())  # -2ab => (a-b)^2
        dist = dist.clamp(min=1e-12).sqrt()  # sqrt  此处dist为input[i] input[j]的欧氏距离

        # For each anchor, locate the input positive case, and find the hardest negative
        mask = labels.expand(n, n).eq(labels.expand(n, n).t())  # dist[i][j]=1 if i&j same label, 0 otherwise
        dist_ap, dist_an = [], []#和对应正例样本的距离，和最难负例样本的距离
        for i in range(batch_size):
            dist_ap.append(dist[i][i+batch_size].unsqueeze(0))  # the input paired positive case
            dist_an.append(dist[i][mask[i]==0].min().unsqueeze(0))  # find hardest negative (diff-label case with the largest distance)
        dist_ap = th.cat(dist_ap)
        dist_an = th.cat(dist_an) # batch_size,

        # Compute ranking hinge loss
        y = th.ones_like(dist_an)
        loss = self.ranking_loss(dist_an, dist_ap, y)#负样本距离，正样本距离，全1向量；当负样本距离全大于正样本距离时，loss=0
        return loss

class TripletLoss_cosine(nn.Module):
    def __init__(self, margin=0.3):
        super(TripletLoss_cosine, self).__init__()
        self.margin = margin
        self.ranking_loss = nn.MarginRankingLoss(margin=margin)
        self.cos_dist = nn.CosineSimilarity(dim=-1)

    def forward(self, inputs, labels):
        ipdb.set_trace()
        n = inputs.size(0)
        batch_size = n // 2
        dist = self.cos_dist(inputs.unsqueeze(1), inputs.unsqueeze(0)) #2bs * 2bs

        # For each anchor, locate the input positive case, and find the hardest negative
        mask = labels.expand(n, n).eq(labels.expand(n, n).t())  # dist[i][j]=1 if i&j same label, 0 otherwise
        dist_ap, dist_an = [], []
        for i in range(batch_size):
            dist_ap.append(dist[i][i+batch_size].unsqueeze(0))  # the input paired positive case
            dist_an.append(dist[i][mask[i]==0].min().unsqueeze(0))  # find hardest negative (diff-label case with the largest distance)
        dist_ap = th.cat(dist_ap)
        dist_an = th.cat(dist_an) # batch_size,

        # Compute ranking hinge loss
        y = th.ones_like(dist_an)
        loss = self.ranking_loss(dist_an, dist_ap, y)
        return loss
    
class SupConLoss(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""
    def __init__(self, temperature=0.1, contrast_mode='all',
                 base_temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, labels=None, mask=None):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf

        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """
        device = (th.device('cuda')
                  if features.is_cuda
                  else th.device('cpu'))

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)
#         import ipdb
#         ipdb.set_trace()

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = th.eye(batch_size, dtype=th.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = th.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]
        contrast_feature = th.cat(th.unbind(features, dim=1), dim=0)
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))
            

        # compute logits
        anchor_dot_contrast = th.div(
            th.matmul(anchor_feature, contrast_feature.T),
            self.temperature)
#         anchor_dot_contrast =th.matmul(anchor_feature, contrast_feature.T)
        
        # for numerical stability
        logits_max, _ = th.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = th.scatter(
            th.ones_like(mask),
            1,
            th.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = th.exp(logits) * logits_mask
        log_prob = logits - th.log(exp_logits.sum(1, keepdim=True) + 1e-6)
#         print(f"logits max{th.max(logits)}\t logits min{th.min(logits)}\t log_prob max{th.max(log_prob)}\t log_prob min{th.min(log_prob)}")

        # compute mean of log-likelihood over positive
#         mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)
        mean_log_prob_pos = (mask*log_prob).sum(1)/(mask.sum(1)+1e-6)

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()
#         import ipdb
#         ipdb.set_trace()

        return loss