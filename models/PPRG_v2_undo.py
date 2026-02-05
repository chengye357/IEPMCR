import os
import torch as th
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from transformers import AutoModel, AutoTokenizer
from pprg_end2end.utils import INT, FLOAT, LONG, cast_type, summary
from pprg_end2end.models import MLPLayer, TripletLoss, create_vit, init_tokenizer, tie_encoder_decoder_weights, BertConfig, BertModel, BertLMHeadModel, BertEncoder
import ipdb
import logging
logger = logging.getLogger()

def cluster_cl(cluster_result):
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
    return proto_logits, proto_labels

class PPRG_Pretrain(nn.Module):
    def __init__(self,                 
                 med_config = 'configs/bert_config.json',  
                 image_size = 224,
                 vit_grad_ckpt = False,
                 vit_ckpt_layer = 0,                    
                 embed_dim = 256,     
                 queue_size = 57600,
                 momentum = 0.995,
                 ):
        """
        Args:
            med_config (str): path for the mixture of encoder-decoder model's configuration file
            image_size (int): input image size
            vit (str): model size of vision transformer
        """               
        super().__init__()
        
        #initial vit and load from google        
        self.visual_encoder, vision_width = create_vit(image_size, vit_grad_ckpt, vit_ckpt_layer, 0)
        checkpoint = torch.hub.load_state_dict_from_url(
            url="https://dl.fbaipublicfiles.com/deit/deit_base_patch16_224-b5f2ef4d.pth",
            map_location="cpu", check_hash=True)
        state_dict = checkpoint["model"]     
        msg = self.visual_encoder.load_state_dict(state_dict,strict=False)      
               
         #initial bert and load from bert-base-uncased   
        self.tokenizer = init_tokenizer()   
        encoder_config = BertConfig.from_json_file(med_config)
        encoder_config.encoder_width = vision_width
        self.text_encoder = BertModel.from_pretrained('bert-base-uncased',config=encoder_config, add_pooling_layer=False)
        self.text_encoder.resize_token_embeddings(len(self.tokenizer)) 

        text_width = self.text_encoder.config.hidden_size
        
        self.vision_proj = nn.Linear(vision_width, embed_dim)
        self.text_proj = nn.Linear(text_width, embed_dim)

#         self.itm_head = nn.Linear(text_width, 2) ###isdel
        
        # create momentum encoders  
        self.visual_encoder_m, vision_width = create_vit(vit,image_size)              
        self.vision_proj_m = nn.Linear(vision_width, embed_dim)
        self.text_encoder_m = BertModel(config=encoder_config, add_pooling_layer=False)      
        self.text_proj_m = nn.Linear(text_width, embed_dim)
        
        self.model_pairs = [[self.visual_encoder,self.visual_encoder_m],
                            [self.vision_proj,self.vision_proj_m],
                            [self.text_encoder,self.text_encoder_m],
                            [self.text_proj,self.text_proj_m],
                           ]       
        self.copy_params()

        # create the queue
        self.register_buffer("image_queue", torch.randn(embed_dim, queue_size))
        self.register_buffer("text_queue", torch.randn(embed_dim, queue_size))
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))  

        self.image_queue = nn.functional.normalize(self.image_queue, dim=0)
        self.text_queue = nn.functional.normalize(self.text_queue, dim=0)
        
        self.queue_size = queue_size
        self.momentum = momentum
        self.temp = nn.Parameter(0.07*torch.ones([]))   
        
        # create crossmodal encoder to retrieve similay case        
#         self.crossmodal_encoder = BertEncoder(config)
        
        # create the decoder
        decoder_config = BertConfig.from_json_file(med_config)
        decoder_config.encoder_width = vision_width        
        self.text_decoder = BertLMHeadModel.from_pretrained('bert-base-uncased',config=decoder_config)    
        self.text_decoder.resize_token_embeddings(len(self.tokenizer)) #####这里我记得需要加token
#         tie_encoder_decoder_weights(self.text_encoder,self.text_decoder.bert,'','/attention')#####共享输入模型的参数

        self.text_mlp = MLPLayer(self.dim_text_in, self.dim_text_out)
        self.img_mlp = MLPLayer(self.dim_img_in, self.dim_img_out)
        self.sim = nn.CosineSimilarity(dim=-1)
        self.loss_fct = TripletLoss(margin=self.margin)
        
    def forward(self, image, caption, alpha):
        with torch.no_grad():
            self.temp.clamp_(0.001,0.5)
        
        image_embeds = self.visual_encoder(image) #     visual embedding  in shape : B N C_768
        image_atts = torch.ones(image_embeds.size()[:-1],dtype=torch.long).to(image.device) #
        image_feat = F.normalize(self.vision_proj(image_embeds[:,0,:]),dim=-1)  #  project cls token into visual vector in shape: B C_768
        
        text = self.tokenizer(caption, padding='max_length', truncation=True, max_length=30, return_tensors="pt").to(image.device) # semantic embedding in shape B N C
        text_output = self.text_encoder(text.input_ids, attention_mask = text.attention_mask, return_dict = True, mode = 'text')            
        text_feat = F.normalize(self.text_proj(text_output.last_hidden_state[:,0,:]),dim=-1)               
 

         #crossmodal_encoder_outputs = self.crossmodal_encoder(
#             text_output,
#             attention_mask=text.attention_mask,
#             encoder_hidden_states=image_embeds,
#             encoder_attention_mask=image_atts,
#             return_dict=True,
#         )

        # get momentum features
        with torch.no_grad():
            self._momentum_update()
            image_embeds_m = self.visual_encoder_m(image) 
            image_feat_m = F.normalize(self.vision_proj_m(image_embeds_m[:,0,:]),dim=-1)  # B Cv
            image_feat_all = torch.cat([image_feat_m.t(),self.image_queue.clone().detach()],dim=1)   #  Cv B+B_m
            
            text_output_m = self.text_encoder_m(text.input_ids, attention_mask = text.attention_mask,return_dict = True, mode = 'text')    
            text_feat_m = F.normalize(self.text_proj_m(text_output_m.last_hidden_state[:,0,:]),dim=-1)   #B Cs
            text_feat_all = torch.cat([text_feat_m.t(),self.text_queue.clone().detach()],dim=1)   #Cs B+Bm
            
            

            
            if use_crossmodal_cl:
                sim_i2t_m = image_feat_m @ text_feat_all / self.temp  # B Cv @  Cs B+Bm  求当前batch的每个图片与队列中全部文本的相似度  means Cv=Cs
                sim_t2i_m = text_feat_m @ image_feat_all / self.temp

                sim_targets = torch.zeros(sim_i2t_m.size()).to(image.device)
                sim_targets.fill_diagonal_(1)          #对角线为1，其余为0 的矩阵

                sim_i2t_targets = alpha * F.softmax(sim_i2t_m, dim=1) + (1 - alpha) * sim_targets       #C B+Bm
                sim_t2i_targets = alpha * F.softmax(sim_t2i_m, dim=1) + (1 - alpha) * sim_targets   
                
        if use_crossmodal_cl: 
            sim_i2t = image_feat @ text_feat_all / self.temp        #B B+Bm
            sim_t2i = text_feat @ image_feat_all / self.temp
            # 对应位置相乘,这个过程中，我们将目标相似度分数的作用视为一个先验知识，以约束当前计算得到的相似度分布接近目标相似度分布
            loss_i2t = -torch.sum(F.log_softmax(sim_i2t, dim=1)*sim_i2t_targets,dim=1).mean()
            loss_t2i = -torch.sum(F.log_softmax(sim_t2i, dim=1)*sim_t2i_targets,dim=1).mean() 

            loss_ita = (loss_i2t+loss_t2i)/2
            
        self._dequeue_and_enqueue(image_feat_m, text_feat_m)             
            
        # compute query features
        l_image_pos = torch.einsum('nc,nc->n', [image_feat, image_feat_m]).unsqueeze(-1)
        l_image_neg = torch.einsum('nc,ck->nk', [image_feat, image_feat_all.clone().detach()])
        logits_image = torch.cat([l_image_pos, l_image_neg], dim=1)
        logits_image /= self.T
        labels_image = torch.zeros(logits_image.shape[0], dtype=torch.long).cuda()    

        l_text_pos = torch.einsum('nc,nc->n', [text_feat, text_feat_m]).unsqueeze(-1)
        l_text_neg = torch.einsum('nc,ck->nk', [text_feat, text_feat_all.clone().detach()])
        logits_text = torch.cat([l_text_pos, l_text_neg], dim=1)
        logits_text /= self.T
        labels_text = torch.zeros(logits_text.shape[0], dtype=torch.long).cuda()  
        
        criterion = nn.CrossEntropyLoss().cuda(args.gpu)
        loss_i2i = criterion(logits_image, labels_image) 
        loss_t2t = criterion(logits_text, labels_text) 
        
        
        text_pair_in = torch.cat([text_feat, ], dim=0)#不对，这里text_pair_in还是相似样本
        img_pair_in = torch.cat([image_feat, ], dim=0)
        #ori triplet ranking loss from RERG
        text_pair_out = self.text_mlp(text_pair_in)  # bs*2, dim_text_out     768 -> 768
        img_pair_out = self.img_mlp(img_pair_in)  # bs*2, dim_text_out      2048 -> 128
        case_out = th.cat([text_pair_out, img_pair_out], dim=1)  # bs*2, dim_multimodal_out  896 -> 896
        loss_tri = self.loss_fct(case_out, labels)
        loss_dict = {'loss_triponline': loss_tri}
        
        # clustering        
#         proto_image_logits, proto_image_labels = cluster_cl(cluster_image_result)
#         proto_text_logits, proto_text_labels = cluster_cl(cluster_text_result)        
        
        #decoder      

    
        
        self.text_decoder
         
        crossmodal_encoder_outputs = self.crossmodal_encoder(
            text_output,
            attention_mask=text.attention_mask,
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=image_atts,
            return_dict=True,
        )        
        ##================= LM ========================##  
        
            
        
        decoder_input_ids = text.input_ids.clone()      
        decoder_input_ids[:,0] = self.tokenizer.bos_token_id
        decoder_targets = decoder_input_ids.masked_fill(decoder_input_ids == self.tokenizer.pad_token_id, -100) 
        #crossmodal decoder
        decoder_output_crossmodal = self.text_decoder(decoder_input_ids, 
                               attention_mask = text.attention_mask, 
                               encoder_hidden_states = image_embeds,
                               encoder_attention_mask = image_atts,                  
                               labels = decoder_targets,
                               return_dict = True,   
                              ) 
        decoder_output_vertical = self.text_decoder(decoder_input_ids, 
                               attention_mask = text.attention_mask, 
                               encoder_hidden_states = curr_rsp_teach_embedding,
                               encoder_attention_mask = curr_rsp_teach_embedding_atts,                  
                               labels = decoder_targets,
                               return_dict = True,   
                              )  
        decoder_output_ = self.text_decoder(decoder_input_ids, 
                               attention_mask = text.attention_mask, 
                               encoder_hidden_states = sim_case_ctx_embedding,
                               encoder_attention_mask = sim_case_ctx_embedding_atts,                  
                               labels = decoder_targets,
                               return_dict = True,   
                              )  
          
        loss_lm = decoder_output.loss                
        return loss_ita, loss_itm, loss_lm
 


    @torch.no_grad()    
    def copy_params(self):
        for model_pair in self.model_pairs:           
            for param, param_m in zip(model_pair[0].parameters(), model_pair[1].parameters()):
                param_m.data.copy_(param.data)  # initialize
                param_m.requires_grad = False  # not update by gradient    

            
    @torch.no_grad()        
    def _momentum_update(self):
        for model_pair in self.model_pairs:           
            for param, param_m in zip(model_pair[0].parameters(), model_pair[1].parameters()):
                param_m.data = param_m.data * self.momentum + param.data * (1. - self.momentum)

                        
    @torch.no_grad()
    def _dequeue_and_enqueue(self, image_feat, text_feat):
        # gather keys before updating queue
        image_feats = concat_all_gather(image_feat)
        text_feats = concat_all_gather(text_feat)

        batch_size = image_feats.shape[0]

        ptr = int(self.queue_ptr)
        assert self.queue_size % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        self.image_queue[:, ptr:ptr + batch_size] = image_feats.T
        self.text_queue[:, ptr:ptr + batch_size] = text_feats.T
        ptr = (ptr + batch_size) % self.queue_size  # move pointer

        self.queue_ptr[0] = ptr 

        
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