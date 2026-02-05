import os
import torch as th
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from transformers import AutoModel, AutoTokenizer
from pprg_end2end.utils import INT, FLOAT, LONG, cast_type, summary
from pprg_end2end.models import MLPLayer, TripletLoss, VisionTransformer, interpolate_pos_embed
import ipdb
import logging
logger = logging.getLogger()
def create_vit(image_size, use_grad_checkpointing=False, ckpt_layer=0, drop_path_rate=0):
    vision_width = 768
    visual_encoder = VisionTransformer(
        img_size=image_size, 
        patch_size=16, 
        embed_dim=vision_width, 
        depth=12, 
        num_heads=12, 
        use_grad_checkpointing=use_grad_checkpointing,
        ckpt_layer=ckpt_layer,
        drop_path_rate=0 or drop_path_rate) 
    return visual_encoder

class VisualPCL(nn.Module):
    """
    Build a MoCo model with: a query encoder, a key encoder, and a queue
    https://arxiv.org/abs/1911.05722
    """
    def __init__(self, base_encoder, dim=128, r=16384, m=0.999, T=0.1, mlp=False):
        """
        dim: feature dimension (default: 128)
        r: queue size; number of negative samples/prototypes (default: 16384)
        m: momentum for updating key encoder (default: 0.999)
        T: softmax temperature 
        mlp: whether to use mlp projection
        """
        super(MoCo, self).__init__()

        self.r = r
        self.m = m
        self.T = T

        # create the encoders
        # num_classes is the output fc dimension
        self.encoder_q = create_vit(image_size, vit_grad_ckpt, vit_ckpt_layer, 0)
        self.encoder_k = create_vit(image_size, vit_grad_ckpt, vit_ckpt_layer, 0)

        if mlp:  # hack: brute-force replacement
            dim_mlp = self.encoder_q.fc.weight.shape[1]
            self.encoder_q.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), self.encoder_q.fc)
            self.encoder_k.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), self.encoder_k.fc)

        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient

        # create the queue
        self.register_buffer("queue", torch.randn(dim, r))
        self.queue = nn.functional.normalize(self.queue, dim=0)

        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """
        Momentum update of the key encoder
        """
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys):
        # gather keys before updating queue
        keys = concat_all_gather(keys)

        batch_size = keys.shape[0]

        ptr = int(self.queue_ptr)
        assert self.r % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        self.queue[:, ptr:ptr + batch_size] = keys.T
        ptr = (ptr + batch_size) % self.r  # move pointer

        self.queue_ptr[0] = ptr

    @torch.no_grad()
    def _batch_shuffle_ddp(self, x):
        """
        Batch shuffle, for making use of BatchNorm.
        *** Only support DistributedDataParallel (DDP) model. ***
        """
        # gather from all gpus
        batch_size_this = x.shape[0]
        x_gather = concat_all_gather(x)
        batch_size_all = x_gather.shape[0]

        num_gpus = batch_size_all // batch_size_this

        # random shuffle index
        idx_shuffle = torch.randperm(batch_size_all).cuda()

        # broadcast to all gpus
        torch.distributed.broadcast(idx_shuffle, src=0)

        # index for restoring
        idx_unshuffle = torch.argsort(idx_shuffle)

        # shuffled index for this gpu
        gpu_idx = torch.distributed.get_rank()
        idx_this = idx_shuffle.view(num_gpus, -1)[gpu_idx]

        return x_gather[idx_this], idx_unshuffle

    @torch.no_grad()
    def _batch_unshuffle_ddp(self, x, idx_unshuffle):
        """
        Undo batch shuffle.
        *** Only support DistributedDataParallel (DDP) model. ***
        """
        # gather from all gpus
        batch_size_this = x.shape[0]
        x_gather = concat_all_gather(x)
        batch_size_all = x_gather.shape[0]

        num_gpus = batch_size_all // batch_size_this

        # restored index for this gpu
        gpu_idx = torch.distributed.get_rank()
        idx_this = idx_unshuffle.view(num_gpus, -1)[gpu_idx]

        return x_gather[idx_this]

    def forward(self, im_q, im_k=None, is_eval=False, cluster_result=None, index=None):
        """
        Input:
            im_q: a batch of query images
            im_k: a batch of key images
            is_eval: return momentum embeddings (used for clustering)
            cluster_result: cluster assignments, centroids, and density
            index: indices for training samples
        Output:
            logits, targets, proto_logits, proto_targets
        """
        
        if is_eval:
            k = self.encoder_k(im_q)  
            k = nn.functional.normalize(k, dim=1)            
            return k
        
        # compute key features
        with torch.no_grad():  # no gradient to keys
            self._momentum_update_key_encoder()  # update the key encoder

            # shuffle for making use of BN
            im_k, idx_unshuffle = self._batch_shuffle_ddp(im_k)

            k = self.encoder_k(im_k)  # keys: NxC
            k = nn.functional.normalize(k, dim=1)

            # undo shuffle
            k = self._batch_unshuffle_ddp(k, idx_unshuffle)

        # compute query features
        q = self.encoder_q(im_q)  # queries: NxC
        q = nn.functional.normalize(q, dim=1)
        
        # compute logits
        # Einstein sum is more intuitive
        # positive logits: Nx1
        l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)
        # negative logits: Nxr
        l_neg = torch.einsum('nc,ck->nk', [q, self.queue.clone().detach()])

        # logits: Nx(1+r)
        logits = torch.cat([l_pos, l_neg], dim=1)

        # apply temperature
        logits /= self.T

        # labels: positive key indicators
        labels = torch.zeros(logits.shape[0], dtype=torch.long).cuda()

        # dequeue and enqueue
        self._dequeue_and_enqueue(k)
        
        # prototypical contrast
        if cluster_result is not None:  
            proto_labels = []
            proto_logits = []
            for n, (im2cluster,prototypes,density) in enumerate(zip(cluster_result['im2cluster'],cluster_result['centroids'],cluster_result['density'])):
                # get positive prototypes
                pos_proto_id = im2cluster[index]  #聚类id
                pos_prototypes = prototypes[pos_proto_id]      #原型id
                
                # sample negative prototypes
                all_proto_id = [i for i in range(im2cluster.max()+1)]       
                neg_proto_id = set(all_proto_id)-set(pos_proto_id.tolist())
                neg_proto_id = sample(neg_proto_id,self.r) #sample r negative prototypes 
                neg_prototypes = prototypes[neg_proto_id]    

                proto_selected = torch.cat([pos_prototypes,neg_prototypes],dim=0)
                
                # compute prototypical logits
                logits_proto = torch.mm(q,proto_selected.t())
                
                # targets for prototype assignment
                labels_proto = torch.linspace(0, q.size(0)-1, steps=q.size(0)).long().cuda()
                
                # scaling temperatures for the selected prototypes
                temp_proto = density[torch.cat([pos_proto_id,torch.LongTensor(neg_proto_id).cuda()],dim=0)]  
                logits_proto /= temp_proto
                
                proto_labels.append(labels_proto)
                proto_logits.append(logits_proto)
            return logits, labels, proto_logits, proto_labels
        else:
            return logits, labels, None, None

class TextPCL(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
    def forward(self, data_feed):
        return loss_dict

class Encoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
    def forward(self, data_feed):
        return loss_dict
    
class Decoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
    def forward(self, data_feed):
        return loss_dict


class CaseEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.use_gpu = config.use_gpu
        self.device = th.device("cuda:0" if  self.use_gpu else "cpu")
        self.margin = config.margin
        self.text_model_name_or_path = config.text_model_name_or_path
        self.text_pooler = config.text_pooler  # 'cls' / 'cls_before_pooler'
        self.max_length = config.max_seq_length
        self.dim_text_in = config.dim_text_in
        self.dim_text_out = config.dim_text_out
        self.dim_img_in = config.dim_img_in
        self.dim_img_out = config.dim_img_out
        self.dim_case_out = self.dim_text_out +  self.dim_img_out

        self.init_net()

    def init_net(self):
        self.text_model = AutoModel.from_pretrained(self.text_model_name_or_path).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(self.text_model_name_or_path)
        self.text_mlp = MLPLayer(self.dim_text_in, self.dim_text_out)
        self.img_mlp = MLPLayer(self.dim_img_in, self.dim_img_out)
        self.sim = nn.CosineSimilarity(dim=-1)
        self.loss_fct = TripletLoss(margin=self.margin)

    def np2var(self, inputs, dtype):
        if inputs is None:
            return None
        return cast_type(Variable(th.from_numpy(inputs)),
                         dtype,
                         self.use_gpu)

    def backward(self, loss, batch_cnt, add_loss_adv=None):
        total_loss = self.valid_loss(loss, batch_cnt, add_loss_adv=add_loss_adv)
        total_loss.backward()

    def valid_loss(self, loss, batch_cnt=None, add_loss_adv=None):
        total_loss = 0.0
        for k, l in loss.items():
            if l is not None:
                total_loss += l
        return total_loss

    def model_sel_loss(self, loss, batch_cnt):
        return self.valid_loss(loss, batch_cnt)

    def get_optimizer(self, config, verbose=True):
        if config.op == 'adam':
            if verbose:
                print('Use Adam')
            return optim.Adam(filter(lambda p: p.requires_grad, self.parameters()), lr=config.init_lr,
                              weight_decay=config.l2_norm)
        elif config.op == 'sgd':
            print('Use SGD')
            return optim.SGD(self.parameters(), lr=config.init_lr, momentum=config.momentum)
        elif config.op == 'rmsprop':
            print('Use RMSProp')
            return optim.RMSprop(self.parameters(), lr=config.init_lr, momentum=config.momentum)

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

    def forward(self, data_feed):
#         ipdb.set_trace()
        text_pair_in_str = data_feed['text_pair_in']  # batch_size * 2
        img_pair_in = self.np2var(data_feed['img_pair_in'], FLOAT)  # batch_size * 2, dim_img_in
        labels = self.np2var(data_feed['labels'], INT)  # batch_size * 2

        text_pair_in_id = self.tokenizer(text_pair_in_str, return_tensors='pt', padding=True,
                                      max_length=self.max_length, truncation=True).to(self.device)#16 182
        text_pair_in_vec = self.text_model(**text_pair_in_id, output_hidden_states=True, return_dict=True)#Bert  odict_keys(['last_hidden_state', 'pooler_output', 'hidden_states'])
        last_hidden = text_pair_in_vec.last_hidden_state#16 182 768
        pooler_output = text_pair_in_vec.pooler_output#16 768

        text_pair_in = None
        if self.text_pooler == 'cls':
            # There is a linear+activation layer after CLS representation
            text_pair_in = pooler_output
        elif self.text_pooler == 'cls_before_pooler':
            text_pair_in = last_hidden[:, 0]

        text_pair_out = self.text_mlp(text_pair_in)  # bs*2, dim_text_out, 768 -> 768
        img_pair_out = self.img_mlp(img_pair_in)  # bs*2, dim_text_out, 2048 -> 128
        case_out = th.cat([text_pair_out, img_pair_out], dim=1)  # 896 -> 896

        loss = self.loss_fct(case_out, labels)
        loss_dict = {'loss_triponline': loss}
        return loss_dict

    def get_case_out(self, data_feed):
        text_in_str = data_feed['text_in']  # batch_size
        img_in = self.np2var(data_feed['img_in'], FLOAT)  # batch_size, dim_img_in

        text_in_id = self.tokenizer(text_in_str, return_tensors='pt', padding=True,
                                    max_length=self.max_length, truncation=True).to(self.device)
        text_in_vec = self.text_model(**text_in_id, output_hidden_states=True, return_dict=True)
        last_hidden = text_in_vec.last_hidden_state
        pooler_output = text_in_vec.pooler_output

        text_in = None
        if self.text_pooler == 'cls':
            # There is a linear+activation layer after CLS representation
            text_in = pooler_output
        elif self.text_pooler == 'cls_before_pooler':
            text_in = last_hidden[:, 0]

        text_out = self.text_mlp(text_in)  # bs, dim_text_out
        img_out = self.img_mlp(img_in)  # bs, dim_text_out
        case_out = th.cat([text_out, img_out], dim=1)

        return case_out  # bs, dim_case_out

    def get_similarity(self, data_feed, train_vecs, topk=(1,), num_cluster_candidates=None):
#         ipdb.set_trace()
        text_in_str = data_feed['text_in'] # batch_size
        img_in = self.np2var(data_feed['img_in'], FLOAT)  # batch_size, dim_img_in

        text_in_id = self.tokenizer(text_in_str, return_tensors='pt', padding=True,
                                    max_length=self.max_length, truncation=True).to(self.device)
        text_in_vec = self.text_model(**text_in_id, output_hidden_states=True, return_dict=True)
        last_hidden = text_in_vec.last_hidden_state
        pooler_output = text_in_vec.pooler_output

        text_in = None
        if self.text_pooler == 'cls':
            # There is a linear+activation layer after CLS representation
            text_in = pooler_output
        elif self.text_pooler == 'cls_before_pooler':
            text_in = last_hidden[:, 0]

        text_out = self.text_mlp(text_in)  # bs, dim_text_out
        img_out = self.img_mlp(img_in)  # bs, dim_text_out
        test_case_out = th.cat([text_out, img_out], dim=1)

        train_vecs_in = self.np2var(train_vecs, FLOAT)  # train_size, dim_case_out  
        cos_sim = self.sim(test_case_out.unsqueeze(1), train_vecs_in.unsqueeze(0))  # bs, train_size   计算与train数据的余弦相似度

        maxk = max(topk)
        topk_values, topk_caseids_out = cos_sim.topk(maxk, 1, True, True)  # bs, K  根据余弦相似度来检索前k个最相似的值和相应的案例ID

        candidate_scores, candidate_caseids = None, None
        if num_cluster_candidates:
            candidate_scores, candidate_caseids = cos_sim.topk(num_cluster_candidates, 1, True, True)  # bs, K  

        return test_case_out, topk_caseids_out, candidate_scores, candidate_caseids

    def get_similarity_train(self, data_feed, train_vecs, topk=(1,), num_cluster_candidates=None):
        text_in_str = data_feed['text_in'] # batch_size
        img_in = self.np2var(data_feed['img_in'], FLOAT)  # batch_size, dim_img_in

        text_in_id = self.tokenizer(text_in_str, return_tensors='pt', padding=True,
                                    max_length=self.max_length, truncation=True).to(self.device)
        text_in_vec = self.text_model(**text_in_id, output_hidden_states=True, return_dict=True)
        last_hidden = text_in_vec.last_hidden_state
        pooler_output = text_in_vec.pooler_output

        text_in = None
        if self.text_pooler == 'cls':
            # There is a linear+activation layer after CLS representation
            text_in = pooler_output
        elif self.text_pooler == 'cls_before_pooler':
            text_in = last_hidden[:, 0]

        text_out = self.text_mlp(text_in)  # bs, dim_text_out
        img_out = self.img_mlp(img_in)  # bs, dim_text_out
        test_case_out = th.cat([text_out, img_out], dim=1)

        train_vecs_in = self.np2var(train_vecs, FLOAT)  # train_size, dim_case_out
        cos_sim = self.sim(test_case_out.unsqueeze(1), train_vecs_in.unsqueeze(0))  # bs, train_size 

        maxk = max(topk) + 1
        topk_values, topk_caseids_out = cos_sim.topk(maxk, 1, True, True)  # bs, K   

        candidate_scores, candidate_caseids = None, None
        if num_cluster_candidates:
            candidate_scores, candidate_caseids = cos_sim.topk(num_cluster_candidates+1, 1, True, True)

        return topk_caseids_out, candidate_scores, candidate_caseids

    def get_similarity_text(self, data_feed, train_vecs, topk=(1,)):
        text_in_vec = data_feed['text_in_vec'] # batch_size

        # text_in_id = self.tokenizer(text_in_str, return_tensors='pt', padding=True,
        #                             max_length=self.max_length, truncation=True).to(self.device)
        # text_in_vec = self.text_model(**text_in_id, output_hidden_states=True, return_dict=True)
        # last_hidden = text_in_vec.last_hidden_state
        # pooler_output = text_in_vec.pooler_output
        #
        # text_in = None
        # if self.text_pooler == 'cls':
        #     # There is a linear+activation layer after CLS representation
        #     text_in = pooler_output
        # elif self.text_pooler == 'cls_before_pooler':
        #     text_in = last_hidden[:, 0]

        # text_out = self.text_mlp(text_in)  # bs, dim_text_out
        # img_out = self.img_mlp(img_in)  # bs, dim_text_out
        # test_case_out = th.cat([text_out, img_out], dim=1)

        train_vecs_in = self.np2var(train_vecs, FLOAT)  # train_size, dim_case_out
        # cos_sim = self.sim(test_case_out.unsqueeze(1), train_vecs_in.unsqueeze(0))  # bs, train_size
        text_in_vec = self.np2var(text_in_vec, FLOAT)
        cos_sim = self.sim(text_in_vec.unsqueeze(1), train_vecs_in.unsqueeze(0))

        maxk = max(topk)
        topk_values, topk_caseids_out = cos_sim.topk(maxk, 1, True, True)  # bs, K

        return topk_caseids_out

    def get_similarity_img(self, data_feed, train_vecs, topk=(1,)):
        # text_in_vec = data_feed['text_in_vec'] # batch_size
        img_in = self.np2var(data_feed['img_in'], FLOAT)

        # text_in_id = self.tokenizer(text_in_str, return_tensors='pt', padding=True,
        #                             max_length=self.max_length, truncation=True).to(self.device)
        # text_in_vec = self.text_model(**text_in_id, output_hidden_states=True, return_dict=True)
        # last_hidden = text_in_vec.last_hidden_state
        # pooler_output = text_in_vec.pooler_output
        #
        # text_in = None
        # if self.text_pooler == 'cls':
        #     # There is a linear+activation layer after CLS representation
        #     text_in = pooler_output
        # elif self.text_pooler == 'cls_before_pooler':
        #     text_in = last_hidden[:, 0]

        # text_out = self.text_mlp(text_in)  # bs, dim_text_out
        # img_out = self.img_mlp(img_in)  # bs, dim_text_out
        # test_case_out = th.cat([text_out, img_out], dim=1)

        train_vecs_in = self.np2var(train_vecs, FLOAT)  # train_size, dim_case_out
        # cos_sim = self.sim(test_case_out.unsqueeze(1), train_vecs_in.unsqueeze(0))  # bs, train_size
        # text_in_vec = self.np2var(text_in_vec, FLOAT)
        # cos_sim = self.sim(text_in_vec.unsqueeze(1), train_vecs_in.unsqueeze(0))

        cos_sim = self.sim(img_in.unsqueeze(1), train_vecs_in.unsqueeze(0))

        maxk = max(topk)
        topk_values, topk_caseids_out = cos_sim.topk(maxk, 1, True, True)  # bs, K

        return topk_caseids_out