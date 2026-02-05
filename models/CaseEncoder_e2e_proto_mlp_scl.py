import os
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import json
from random import sample
from torch.autograd import Variable
from transformers import AutoModel, AutoTokenizer
from pprg_end2end.utils import INT, FLOAT, LONG, cast_type, summary
from pprg_end2end.case_util  import EOS, PAD, BOS
from pprg_end2end.models import MLPLayer, TripletLoss, SupConLoss
from pprg_end2end.enc2dec import CtxEncoder, RspEncoder, RspDecoder,CtxEncoder_2, RspEncoder_2
from pprg_end2end.criterions import Loss, NLLEntropy
from pprg_end2end.models import BaseModel
from pprg_end2end.utils import Pack, DATA_PATH
import ipdb
import logging
logger = logging.getLogger()

  
class CaseEncoder_e2e_proto_mlp_scl(BaseModel):
    def __init__(self,corpus,config):
        super(CaseEncoder_e2e_proto_mlp_scl, self).__init__(config)             
        
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
        self.dim_case_out = self.dim_text_out + self.dim_img_out
        
#         vocab_file = DATA_PATH + '/vocab.json'
#         with open(vocab_file, "r") as f:
#             self.vocab_dict = json.load(f)
#             self.vocab = list(self.vocab_dict.keys())
#         self.unk_id = self.vocab_dict['<unk>']
        
#         self.vocab = corpus.vocab
#         self.vocab_dict = corpus.vocab_dict
#         self.vocab_size = len(self.vocab)
#         self.bos_id = self.vocab_dict[BOS]
#         self.eos_id = self.vocab_dict[EOS]
        self.pad_id = 0
        vocab_file = DATA_PATH + '/vocab_bert.json'
#         vocab_file = DATA_PATH + '/vocab.json'
        with open(vocab_file, "r") as f:
            self.vocab_dict = json.load(f)
            self.vocab = list(self.vocab_dict.keys())
            self.vocab_size = len(self.vocab)
        
        
        self.embedding_dim = config.embed_size
        self.hidden_dim = self.config.dec_cell_size
        self.num_sim = self.config.num_sim  
        self.init_net()
        
    def sent2id(self, sent):
        return [self.vocab_dict.get(t, self.unk_id) for t in sent]
    
    def init_net(self):
#         ipdb.set_trace()

        self.text_mlp = MLPLayer(self.dim_text_in, self.dim_text_out)
        self.img_mlp = MLPLayer(self.dim_img_in, self.dim_img_out)
        self.multi_mlp = MLPLayer(self.dim_case_out, self.dim_case_out)
        self.sim = nn.CosineSimilarity(dim=-1)
        self.loss_fct = TripletLoss(margin=self.margin)
        
        self.embedding = nn.Embedding(num_embeddings=self.vocab_size, embedding_dim=self.embedding_dim)
        self.ctx_encoder = CtxEncoder(vocab_size=self.vocab_size,
                                     embedding_dim=self.embedding_dim,
                                     rnn_cell=self.config.enc_rnn_cell,  # gru
                                     hidden_size=self.config.enc_cell_size,
                                     num_layers=self.config.num_layers,
                                     input_dropout_p=self.config.dropout,
                                     output_dropout_p=self.config.dropout,
                                     bidirectional=self.config.bi_enc_cell,
                                     variable_lengths=False,  # since all inputs have been padded
                                     use_attn=self.config.ctx_enc_use_attn,
                                     embedding=self.embedding)
        self.rsp_encoder = RspEncoder(vocab_size=self.vocab_size,
                                      embedding_dim=self.embedding_dim,
                                      rnn_cell=self.config.enc_rnn_cell,  # gru
                                      hidden_size=self.config.enc_cell_size,
                                      num_layers=self.config.num_layers,
                                      input_dropout_p=self.config.dropout,
                                      output_dropout_p=self.config.dropout,
                                      bidirectional=self.config.bi_enc_cell,
                                      variable_lengths=False,  # since all inputs have been padded
                                      use_attn=self.config.rsp_enc_use_attn,
                                      embedding=self.embedding)
        self.decoder = RspDecoder(input_dim=self.embedding_dim,
                                       hidden_dim=self.config.dec_cell_size,
                                       vocab_size=self.vocab_size,
                                       use_coverage=self.config.use_coverage)
        self.projection = nn.Linear(self.hidden_dim, self.vocab_size, bias=True)
        # Compute generation probability p_gen with context
        self.w_h_vertical = nn.Linear(self.hidden_dim * 2, 1, bias=False)
        self.w_s_vertical = nn.Linear(self.hidden_dim, 1, bias=False)
        self.w_x_vertical = nn.Linear(self.embedding_dim, 1, bias=True)
        # Compute generation probability p_gen with sim rsp
        self.w_h_horizontal = nn.Linear(self.hidden_dim * 2, 1, bias=False)
        self.w_s_horizontal = nn.Linear(self.hidden_dim, 1, bias=False)
        self.w_x_horizontal = nn.Linear(self.embedding_dim, 1, bias=True)
        # for compute context similarity
        self.sim_score = nn.CosineSimilarity(dim=-1)

        self.relu = nn.ReLU()
        self.criterion = Loss(pad_id=self.pad_id, cov_weight=self.config.cov_weight, use_coverage=self.config.use_coverage)
        self.celoss = nn.CrossEntropyLoss().cuda()
        # self.nll = NLLEntropy(self.pad_id)

        
    def accuracy(self, output, target, topk=(1,)):
        """Computes the accuracy over the k top predictions for the specified values of k"""
        with th.no_grad():
#             ipdb.set_trace()
            maxk = max(topk)
            batch_size = target.size(0)

            _, pred = output.topk(maxk, 1, True, True)
            pred = pred.t()
            correct = pred.eq(target.view(1, -1).expand_as(pred))

            res = []
            for k in topk:
                correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
                res.append(correct_k.mul_(100.0 / batch_size))
            return res

    def forward(self, data_feed, cluster_result=None):       
        batch_size =data_feed['batch_size']
#         d_keys = data_feed.keys()
        d_keys_float32 = ['text_pair_in', 'img_pair_in', 'curr_ctx_feature', 'curr_rsp_feature', 'curr_img', 'curr_rsps_teach_tokenid', 'curr_rsps_target_tokenid', 'sim_ctx_feature', 'sim_rsp_feature', 'sim_imgs']
        d_keys_int32 = ['topk_caseids_in','labels', 'curr_ctx_tokenmask',  'curr_rsp_tokenmask',  'sim_ctx_tokenmask', 'sim_rsp_tokenmask', 'sim_ids', 'batch_size']
        d_keys_int64 = ['curr_rsp_tokenid', 'curr_ctx_tokenid', 'img_labels', 'caseid', 'curr_rsps_teach_tokenid', 'curr_rsps_target_tokenid', 'sim_rsp_tokenid', 'sim_ctx_tokenid']
        
        for i in data_feed.keys():
            before_dtype = data_feed[i].dtype
            before_device = data_feed[i].device
            if i in d_keys_int32:
                data_feed[i] = self.tensor2cuda(data_feed[i],INT)
            elif i in d_keys_int64:
                data_feed[i] = self.tensor2cuda(data_feed[i],LONG)
            else:
                data_feed[i] = self.tensor2cuda(data_feed[i],FLOAT)

                
        loss_result = {}
        ipdb.set_trace() 
        num_sim = self.num_sim
        
        #load data
        text_pair_in = data_feed['text_pair_in']
        img_pair_in = data_feed['img_pair_in']
        labels = data_feed['labels']
        img_labels = data_feed['img_labels']

        caseid = data_feed['caseid']
        curr_ctx_feature = data_feed['curr_ctx_feature']
        curr_ctx_tokenid = data_feed['curr_ctx_tokenid']
        curr_ctx_tokenmask = data_feed['curr_ctx_tokenmask']
#         curr_rsp_feature = data_feed['curr_rsp_feature']
#         curr_rsp_tokenid = data_feed['curr_rsp_tokenid']
#         curr_rsp_tokenmask = data_feed['curr_rsp_tokenmask']
        curr_img = data_feed['curr_img']

        curr_rsps_teach_tokenid = data_feed['curr_rsps_teach_tokenid']
        curr_rsps_target_tokenid = data_feed['curr_rsps_target_tokenid']

        sim_ctx_feature = data_feed['sim_ctx_feature']
#         sim_ctx_tokenid = data_feed['sim_ctx_tokenid']
        sim_ctx_tokenmask = data_feed['sim_ctx_tokenmask']
        sim_rsp_feature = data_feed['sim_rsp_feature']
        sim_rsp_tokenid = data_feed['sim_rsp_tokenid']
        sim_rsp_tokenmask = data_feed['sim_rsp_tokenmask']
    
        sim_ids = data_feed['sim_ids']
#         sim_imgs = data_feed['sim_imgs']

        #retrieval
        text_pair_mid = text_pair_in[:, 0]
        text_pair_out = self.text_mlp(text_pair_mid)  # bs*2, dim_text_out     768 -> 768
        img_pair_out = self.img_mlp(img_pair_in)  # bs*2, dim_text_out      2048 -> 128
        case_in = th.cat([text_pair_out, img_pair_out], dim=1)  # 896 -> 896  
        case_out = self.multi_mlp(case_in) # 896 -> 896 
        
        loss = self.loss_fct(case_out, labels)
        loss_result['loss_triponline'] = loss
        
        
        batch_case = case_out[:batch_size, :]
        
        ipdb.set_trace()
        # supervised contrastive learning
        loss_result['loss_SupConLoss'] = 1e-5 * loss
        if min(img_labels) != max(img_labels):
            ori_img_data = th.reshape(img_pair_out[:batch_size], (batch_size,1,img_pair_out.size(-1)))
            pos_img_data = th.reshape(img_pair_out[batch_size:], (batch_size,1,img_pair_out.size(-1)))
            scl_img_data = th.cat([ori_img_data, pos_img_data],dim=1)
            criterion_scl = SupConLoss()        
#             ipdb.set_trace()
            loss_scl = criterion_scl(scl_img_data, img_labels)
#             print("scl_img_data:", scl_img_data[:, :, 0]) 
#             print("img_labels:", img_labels) 
#             print("loss_scl:", loss_scl) 
            loss_result['loss_SupConLoss'] = 0.1 * loss_scl
#             
#             if(loss_scl.item()>10): loss_result['loss_SupConLoss'] = 0.1 * loss_scl
   
        
#         print(loss_result)
        # image ce loss
#         curr_img_out = self.img_mlp(curr_img)
#         loss_imgce = self.celoss(curr_img_out, img_labels)
#         loss_result['loss_imgce'] = loss_imgce
        
        
        
        
        #prototypical contrastive learning
        if cluster_result is not None:  
            proto_labels = []
            proto_logits = []
            for n, (im2cluster,prototypes,density) in enumerate(zip(cluster_result['im2cluster'],cluster_result['centroids'],cluster_result['density'])):
                
#                 ipdb.set_trace()
                
                # get positive prototypes
                pos_proto_id = im2cluster[data_feed.caseid] #当前batch样本的簇id
                pos_prototypes = prototypes[pos_proto_id]  #当前batch样本的正原型表征
                
                # sample negative prototypes
                all_proto_id = [i for i in range(im2cluster.max()+1)] #所有簇id      
                neg_proto_id = set(all_proto_id)-set(pos_proto_id.tolist()) #负簇id
                neg_proto_id = sample(neg_proto_id,batch_size) #sample r negative prototypes   选self.r个负样本
                neg_prototypes = prototypes[neg_proto_id]   #当前batch样本的负原型表征

                proto_selected = th.cat([pos_prototypes,neg_prototypes],dim=0)
                
                # compute prototypical logits
                logits_proto = th.mm(batch_case, proto_selected.t())#矩阵点乘，每行表示当前样本与所有原型的相似度，对角线表示当前样本与当前正原型的相似度。
                
                # targets for prototype assignment
                labels_proto = th.linspace(0, batch_case.size(0)-1, steps=batch_case.size(0)).long().cuda() #0-31
                
                # scaling temperatures for the selected prototypes
                temp_proto = density[th.cat([pos_proto_id,th.LongTensor(neg_proto_id).cuda()],dim=0)] # 正+负原型的密度
                logits_proto /= temp_proto
                
                proto_labels.append(labels_proto)
                proto_logits.append(logits_proto)
                

            
            loss_proto = 0
            for proto_out,proto_target in zip(proto_logits, proto_labels):
                loss_proto += self.celoss(proto_out, proto_target)  #优化目标：对角线（即当前样本与当前正原型相似度）最大，拉远与其他原型（batch内其他正原型+给定负原型）距离
                accp = self.accuracy(proto_out, proto_target)[0] 
#                 acc_proto.update(accp[0], images[0].size(0))

            # average loss across all sets of prototypes
            loss_proto /= len(self.config.num_cluster) 
            loss_result['loss_pcl'] = loss_proto          
        
        
#         ipdb.set_trace()
        #reuse
        ctx_embedded, ctx_enc_hidden, ctx_enc_pad_mask = self.ctx_encoder(curr_ctx_feature, curr_ctx_tokenmask) 
        sim_ctx_embedded, sim_ctx_enc_hidden, sim_ctx_enc_pad_mask = self.ctx_encoder(sim_ctx_feature, sim_ctx_tokenmask)        
        sim_rsp_embedded, sim_rsp_enc_hidden, sim_rsp_enc_pad_mask = self.rsp_encoder(sim_rsp_feature, sim_rsp_tokenmask) 

        # compute sim rsp copy weights
        k_ctx_embedded = ctx_embedded.unsqueeze(1).repeat(1, num_sim, 1).reshape(batch_size * num_sim, 2*self.hidden_dim)
        scores = self.sim_score(k_ctx_embedded, sim_ctx_embedded)
        scores = scores.reshape(batch_size, num_sim)
        scores = F.softmax(scores, dim=-1) # b, k  scores means γ_k

        # decoding
        # cross-entropy (negative log-likelihood) loss - Eq. (6)
        horizontal_dists = []
        all_pred_words = []

        # Initialize decoder inputs
        dec_emb = self.embedding(curr_rsps_teach_tokenid)  # [B x T x E]
#         dec_emb = curr_rsps_teach_feature # [B x T x E]

        hidden = ctx_embedded[:, :self.hidden_dim] + ctx_embedded[:, self.hidden_dim:] # [B, H]
        hidden = hidden.unsqueeze(0) # [1, B, H]
        k_hidden = hidden.squeeze(0).unsqueeze(1).repeat(1, num_sim, 1).reshape(batch_size * num_sim,self.hidden_dim).unsqueeze(0)#初始解码器状态,信息来自curr_rsps_teach

        for t in range(self.config.max_dec_len):
            ipdb.set_trace() 
            input_t = dec_emb[:, t, :].unsqueeze(1) # b, 1, e  # Decoder input at this timestep
            k_input_t = input_t.repeat(1,num_sim,1).reshape(batch_size*num_sim, -1).unsqueeze(1)
            #输出注意力分布,编码器上下文向量,当前step的隐状态
            context_attn_dist, context_vec, hidden = self.decoder(dec_input=input_t,#当前时间步的decoder输入
                                            prev_h=hidden,#上一时间步的decoder隐藏状态
                                            ctx_enc_hidden=ctx_enc_hidden,#encoder的最后输出
                                            ctx_enc_pad_mask=ctx_enc_pad_mask,#encoder的mask
                                            coverage=None)
            vocab_dist = self.projection(hidden.squeeze(0)) # b, v
            vocab_dist = F.softmax(vocab_dist, dim=-1)

            # Eq. (8) - Compute generation probability p_gen
            context_feat_vertical = self.w_h_vertical(context_vec)  # [B x 1]
            decoder_feat_vertical = self.w_s_vertical(hidden.squeeze(0))  # [B x 1]
            input_feat_vertical = self.w_x_vertical(input_t.squeeze(1))  # [B x 1]
            gen_feat_vertical = context_feat_vertical + decoder_feat_vertical + input_feat_vertical
            p_gen_vertical = th.sigmoid(gen_feat_vertical)  # [B x 1]

            # Eq. (9) - Compute prob dist'n over extended vocabulary
            vocab_dist = p_gen_vertical * vocab_dist  # [B x V]
            weighted_context_attn_dist = (1.0 - p_gen_vertical) * context_attn_dist  # [B x L]
            vertical_dist = vocab_dist.scatter_add(dim=1, index=curr_ctx_tokenid, src=weighted_context_attn_dist)#??

            # horizontal copy
            sim_rsp_attn_dist, sim_rsp_vec, k_hidden = self.decoder.forward_simrsp(dec_input=k_input_t,
                                          prev_h=k_hidden,
                                          sim_rsp_enc_hidden=sim_rsp_enc_hidden,
                                          sim_rsp_enc_pad_mask=sim_rsp_enc_pad_mask,
                                          coverage=None)

            # Eq. (8) - Compute generation probability p_gen
            sim_rsp_feat_horizontal = self.w_h_horizontal(sim_rsp_vec)  # [BK x 1]
            decoder_feat_horizontal = self.w_s_horizontal(k_hidden.squeeze(0))  # [BK x 1]
            input_feat_horizontal = self.w_x_horizontal(k_input_t.squeeze(1))  # [BK x 1]
            gen_feat_horizontal = sim_rsp_feat_horizontal + decoder_feat_horizontal + input_feat_horizontal
            p_gen_horizontal = th.sigmoid(gen_feat_horizontal)  # [BK x 1]

            p_gen_horizontal = p_gen_horizontal.reshape(batch_size, num_sim, 1) # b, k, 1
            sim_rsp_attn_dist = sim_rsp_attn_dist.reshape(batch_size, num_sim, -1) # b, k, L'
            sim_rsps_stacked = sim_rsp_tokenid.reshape(batch_size, num_sim, -1) # b, k, L'
            sim_rsp_copy_weights = scores.unsqueeze(-1) # b, k, 1

            # horizontal_dist = p_gen_horizontal * vertical_dist # b, v
            weighted_p_gen_horizontal = th.bmm(scores.unsqueeze(1), p_gen_horizontal).squeeze(1) # b, 1
            horizontal_dist = weighted_p_gen_horizontal * vertical_dist # b,v  逐元素乘积

            for k in range(num_sim):
                _p_gen_horizontal = p_gen_horizontal[:, k, :] # b,1
                _sim_rsp_attn_dist = sim_rsp_attn_dist[:, k, :] # b, L'
                _sim_rsp_copy_weights = sim_rsp_copy_weights[:, k, :] # b,1
                _sim_rsp = sim_rsps_stacked[:, k, :] # b, L'
                _weighted_sim_rsp_attn_dist = _sim_rsp_copy_weights * (1.0 - _p_gen_horizontal) * _sim_rsp_attn_dist
                horizontal_dist = horizontal_dist.scatter_add(dim=-1, index=_sim_rsp, src=_weighted_sim_rsp_attn_dist)

            # Save outputs for loss computation
            horizontal_dists.append(horizontal_dist)
                     
            _, pred_word = horizontal_dist.topk(1) # b,1
            all_pred_words.append(pred_word.clone())

            
        final_dists = th.stack(horizontal_dists, dim=-1)  # [B x V_x x T]   增加一维
        all_pred_words = th.cat(all_pred_words, dim=1).squeeze() # b, L'   32 50 预测的词id
        nll_loss = self.criterion.nll_loss(final_dists, curr_rsps_target_tokenid)
        loss_result['nll_loss'] = nll_loss
#         print(loss_result)
        ipdb.set_trace()
        return loss_result


    
    def inference(self, data_feed):
        ipdb.set_trace()        
        batch_size =data_feed['batch_size']
#         d_keys = data_feed.keys()
        d_keys_float32 = ['curr_ctx_feature', 'curr_rsp_feature', 'curr_img', 'curr_rsps_teach_tokenid', 'curr_rsps_target_tokenid', 'sim_ctx_feature', 'sim_rsp_feature', 'sim_imgs']
        d_keys_int32 = ['topk_caseids_in','caseid', 'curr_ctx_tokenmask',  'curr_rsp_tokenmask',  'sim_ctx_tokenmask', 'sim_rsp_tokenmask', 'sim_ids', 'batch_size']
        d_keys_int64 = ['curr_rsp_tokenid', 'curr_ctx_tokenid', 'curr_rsps_teach_tokenid', 'curr_rsps_target_tokenid', 'sim_rsp_tokenid', 'sim_ctx_tokenid']
        
        for i in data_feed.keys():
            before_dtype = data_feed[i].dtype
            if i in d_keys_int32:
                data_feed[i] = self.tensor2cuda(data_feed[i],INT)
            elif i in d_keys_int64:
                data_feed[i] = self.tensor2cuda(data_feed[i],LONG)
            else:
                data_feed[i] = self.tensor2cuda(data_feed[i],FLOAT)
#             print(f'key:{i}\t before_dtype:{before_dtype}\t after_dtype:{data_feed[i].dtype}')
                
        loss_result = {}

        num_sim = self.num_sim

        text_pair_in = data_feed['text_pair_in']
        img_pair_in = data_feed['img_pair_in']
        labels = data_feed['labels']

        caseid = data_feed['caseid']
        curr_ctx_feature = data_feed['curr_ctx_feature']
        curr_ctx_tokenid = data_feed['curr_ctx_tokenid']
        curr_ctx_tokenmask = data_feed['curr_ctx_tokenmask']
#         curr_rsp_feature = data_feed['curr_rsp_feature']
#         curr_rsp_tokenid = data_feed['curr_rsp_tokenid']
#         curr_rsp_tokenmask = data_feed['curr_rsp_tokenmask']
        curr_img = data_feed['curr_img']

        curr_rsps_teach_tokenid = data_feed['curr_rsps_teach_tokenid']
        curr_rsps_target_tokenid = data_feed['curr_rsps_target_tokenid']

        sim_ctx_feature = data_feed['sim_ctx_feature']
#         sim_ctx_tokenid = data_feed['sim_ctx_tokenid']
        sim_ctx_tokenmask = data_feed['sim_ctx_tokenmask']
        sim_rsp_feature = data_feed['sim_rsp_feature']
        sim_rsp_tokenid = data_feed['sim_rsp_tokenid']
        sim_rsp_tokenmask = data_feed['sim_rsp_tokenmask']
    
        sim_ids = data_feed['sim_ids']
#         sim_imgs = data_feed['sim_imgs']

        #reuse
        ctx_embedded, ctx_enc_hidden, ctx_enc_pad_mask = self.ctx_encoder(curr_ctx_feature, curr_ctx_tokenmask) 
        sim_ctx_embedded, sim_ctx_enc_hidden, sim_ctx_enc_pad_mask = self.ctx_encoder(sim_ctx_feature, sim_ctx_tokenmask)        
        sim_rsp_embedded, sim_rsp_enc_hidden, sim_rsp_enc_pad_mask = self.rsp_encoder(sim_rsp_feature, sim_rsp_tokenmask) 
        
        # compute sim rsp copy weights
        k_ctx_embedded = ctx_embedded.unsqueeze(1).repeat(1, num_sim, 1).reshape(batch_size * num_sim, 2*self.hidden_dim)
        scores = self.sim_score(k_ctx_embedded, sim_ctx_embedded)
        scores = scores.reshape(batch_size, num_sim)
        scores = F.softmax(scores, dim=-1) # b, k

        # decoding
        # cross-entropy (negative log-likelihood) loss - Eq. (6)
        horizontal_dists = []
        all_pred_words = []

        # Initialize decoder inputs
        dec_emb = self.embedding(curr_rsps_teach_tokenid)
#         dec_emb = curr_rsps_teach_feature  # [B x T x E] 32 50 100
        hidden = ctx_embedded[:, :self.hidden_dim] + ctx_embedded[:, self.hidden_dim:] # [B, H] 32 300
        hidden = hidden.unsqueeze(0) # [1, B, H]
        k_hidden = hidden.squeeze(0).unsqueeze(1).repeat(1, num_sim, 1).reshape(batch_size * num_sim,self.hidden_dim).unsqueeze(0) #1 160 300
        input_t = dec_emb[:, 0, :].unsqueeze(1) #32 1 100

        for t in range(self.config.max_dec_len):

            k_input_t = input_t.repeat(1, num_sim, 1).reshape(batch_size * num_sim, -1).unsqueeze(1)
            #32 131 ; 32 600;  1 32 300
            context_attn_dist, context_vec, hidden = self.decoder(dec_input=input_t,
                                            prev_h=hidden,
                                          ctx_enc_hidden=ctx_enc_hidden,
                                          ctx_enc_pad_mask=ctx_enc_pad_mask,
                                            coverage=None)
            vocab_dist = self.projection(hidden.squeeze(0)) # b, v   .curr_rsps_teach --emb-> dec_emb --dec-> hidden
            vocab_dist = F.softmax(vocab_dist, dim=-1)

            # Eq. (8) - Compute generation probability p_gen
            context_feat_vertical = self.w_h_vertical(context_vec)  # [B x 1]
            decoder_feat_vertical = self.w_s_vertical(hidden.squeeze(0))  # [B x 1]
            input_feat_vertical = self.w_x_vertical(input_t.squeeze(1))  # [B x 1]
            gen_feat_vertical = context_feat_vertical + decoder_feat_vertical + input_feat_vertical
            p_gen_vertical = th.sigmoid(gen_feat_vertical)  # [B x 1]  p_gen_vertical表示阿尔法αto combine the two distributions together

            # Eq. (9) - Compute prob dist'n over extended vocabulary
            vocab_dist = p_gen_vertical * vocab_dist  # [B x V]
            weighted_context_attn_dist = (1.0 - p_gen_vertical) * context_attn_dist  # [B x L]
            #vertical_dist 表示 𝑃𝑚𝑖𝑑𝑑𝑙e_t
            vertical_dist = vocab_dist.scatter_add(dim=-1, index=curr_ctx_tokenid, src=weighted_context_attn_dist)

            # horizontal copy
            sim_rsp_attn_dist, sim_rsp_vec, k_hidden = self.decoder.forward_simrsp(dec_input=k_input_t,
                                      prev_h=k_hidden,
                                      sim_rsp_enc_hidden=sim_rsp_enc_hidden,
                                      sim_rsp_enc_pad_mask=sim_rsp_enc_pad_mask,
                                      coverage=None)

            # Eq. (8) - Compute generation probability p_gen
            sim_rsp_feat_horizontal = self.w_h_horizontal(sim_rsp_vec)  # [BK x 1]
            decoder_feat_horizontal = self.w_s_horizontal(k_hidden.squeeze(0))  # [BK x 1]
            input_feat_horizontal = self.w_x_horizontal(k_input_t.squeeze(1))  # [BK x 1]
            gen_feat_horizontal = sim_rsp_feat_horizontal + decoder_feat_horizontal + input_feat_horizontal
            p_gen_horizontal = th.sigmoid(gen_feat_horizontal)  # [BK x 1]

            p_gen_horizontal = p_gen_horizontal.reshape(batch_size, num_sim, 1) # b, k, 1
            sim_rsp_attn_dist = sim_rsp_attn_dist.reshape(batch_size, num_sim, -1) # b, k, L'
            sim_rsps_stacked = sim_rsp_tokenid.reshape(batch_size, num_sim, -1) # b, k, L'
            sim_rsp_copy_weights = scores.unsqueeze(-1) # b, k, 1

            # horizontal_dist = p_gen_horizontal * vertical_dist # b, v
            weighted_p_gen_horizontal = th.bmm(scores.unsqueeze(1), p_gen_horizontal).squeeze(1)  # b, 1
            horizontal_dist = weighted_p_gen_horizontal * vertical_dist  # b,v   32 9698

            for k in range(num_sim):
                _p_gen_horizontal = p_gen_horizontal[:, k, :] # b,1
                _sim_rsp_attn_dist = sim_rsp_attn_dist[:, k, :] # b, L'
                _sim_rsp_copy_weights = sim_rsp_copy_weights[:, k, :] # b,1
                _sim_rsp = sim_rsps_stacked[:, k, :] # b, L'
                _weighted_sim_rsp_attn_dist = _sim_rsp_copy_weights * (1.0 - _p_gen_horizontal) * _sim_rsp_attn_dist
                horizontal_dist = horizontal_dist.scatter_add(dim=-1, index=_sim_rsp, src=_weighted_sim_rsp_attn_dist)

            # Save outputs for loss computation
            horizontal_dists.append(horizontal_dist)

            # generate  next time decoder input
            _, pred_word = horizontal_dist.topk(1) # b,1
            # pred_word = th.argmax(horizontal_dist, dim=1)
            all_pred_words.append(pred_word.clone())
            input_t = self.embedding(pred_word) # b, 1, e

#         ipdb.set_trace()
        final_dists = th.stack(horizontal_dists, dim=-1)  # [B x V_x x T]
        all_pred_words = th.cat(all_pred_words, dim=1).squeeze() # b, L'   32 50 预测的词id
        nll_loss = self.criterion.nll_loss(final_dists, curr_rsps_target_tokenid)
        loss_result['nll_loss'] = nll_loss

        return loss_result, all_pred_words, curr_rsps_target_tokenid


    def get_case_out(self, data_feed):      
        batch_size =data_feed['batch_size']
        d_keys_float32 = [ 'curr_ctx_feature', 'curr_rsp_feature', 'curr_img', 'curr_rsps_teach_tokenid', 'curr_rsps_target_tokenid', 'sim_ctx_feature', 'sim_rsp_feature', 'sim_imgs']
        d_keys_int32 = ['topk_caseids_in','caseid', 'curr_ctx_tokenmask',  'curr_rsp_tokenmask',  'sim_ctx_tokenmask', 'sim_rsp_tokenmask', 'sim_ids', 'batch_size']
        d_keys_int64 = ['curr_rsp_tokenid', 'curr_ctx_tokenid', 'curr_rsps_teach_tokenid', 'curr_rsps_target_tokenid', 'sim_rsp_tokenid', 'sim_ctx_tokenid']
        
        for i in data_feed.keys():
            before_dtype = data_feed[i].dtype
            if i in d_keys_int32:
                data_feed[i] = self.tensor2cuda(data_feed[i],INT)
            elif i in d_keys_int64:
                data_feed[i] = self.tensor2cuda(data_feed[i],LONG)
            else:
                data_feed[i] = self.tensor2cuda(data_feed[i],FLOAT)

        curr_ctx_feature = data_feed.curr_ctx_feature
        curr_img = data_feed.curr_img

        #retrieval
        text_in = curr_ctx_feature[:, 0]
        text_out = self.text_mlp(text_in)  # bs*2, dim_text_out     768 -> 768
        img_out = self.img_mlp(curr_img)  # bs*2, dim_text_out      2048 -> 128
        case_in = th.cat([text_out, img_out], dim=1)
        case_out = self.multi_mlp(case_in) # 896 -> 896 

        return case_out  # bs, dim_case_out

    def get_similarity(self, data_feed, train_vecs, topk=(1,), num_cluster_candidates=None):
        batch_size =data_feed['batch_size']
        d_keys_float32 = [ 'curr_ctx_feature', 'curr_rsp_feature', 'curr_img', 'curr_rsps_teach_tokenid', 'curr_rsps_target_tokenid', 'sim_ctx_feature', 'sim_rsp_feature', 'sim_imgs']
        d_keys_int32 = ['topk_caseids_in','caseid', 'curr_ctx_tokenmask',  'curr_rsp_tokenmask',  'sim_ctx_tokenmask', 'sim_rsp_tokenmask', 'sim_ids', 'batch_size']
        d_keys_int64 = ['curr_rsp_tokenid', 'curr_ctx_tokenid', 'curr_rsps_teach_tokenid', 'curr_rsps_target_tokenid', 'sim_rsp_tokenid', 'sim_ctx_tokenid']
        
        for i in data_feed.keys():
            before_dtype = data_feed[i].dtype
            if i in d_keys_int32:
                data_feed[i] = self.tensor2cuda(data_feed[i],INT)
            elif i in d_keys_int64:
                data_feed[i] = self.tensor2cuda(data_feed[i],LONG)
            else:
                data_feed[i] = self.tensor2cuda(data_feed[i],FLOAT)

        curr_ctx_feature = data_feed.curr_ctx_feature
        curr_img = data_feed.curr_img

        #retrieval
        text_in = curr_ctx_feature[:, 0]
        text_out = self.text_mlp(text_in)  # bs*2, dim_text_out     768 -> 768
        img_out = self.img_mlp(curr_img)  # bs*2, dim_text_out      2048 -> 128
        case_in = th.cat([text_out, img_out], dim=1)
        test_case_out = self.multi_mlp(case_in) # 896 -> 896 

        train_vecs_in = self.np2var(train_vecs, FLOAT)  # train_size, dim_case_out  
        cos_sim = self.sim(test_case_out.unsqueeze(1), train_vecs_in.unsqueeze(0))  # bs, train_size   计算与train数据的余弦相似度

        maxk = max(topk)
        topk_values, topk_caseids_out = cos_sim.topk(maxk, 1, True, True)  # bs, K  根据余弦相似度来检索前k个最相似的值和相应的案例ID
        candidate_scores, candidate_caseids = None, None
        if num_cluster_candidates:
            candidate_scores, candidate_caseids = cos_sim.topk(num_cluster_candidates, 1, True, True)  # bs, K  

        return test_case_out, topk_caseids_out, candidate_scores, candidate_caseids

    def get_similarity_train(self, data_feed, train_vecs, topk=(1,), num_cluster_candidates=None):
        batch_size =data_feed['batch_size']
        d_keys_float32 = [ 'curr_ctx_feature', 'curr_rsp_feature', 'curr_img', 'curr_rsps_teach_tokenid', 'curr_rsps_target_tokenid', 'sim_ctx_feature', 'sim_rsp_feature', 'sim_imgs']
        d_keys_int32 = ['topk_caseids_in','caseid', 'curr_ctx_tokenmask',  'curr_rsp_tokenmask',  'sim_ctx_tokenmask', 'sim_rsp_tokenmask', 'sim_ids', 'batch_size']
        d_keys_int64 = ['curr_rsp_tokenid', 'curr_ctx_tokenid', 'curr_rsps_teach_tokenid', 'curr_rsps_target_tokenid', 'sim_rsp_tokenid', 'sim_ctx_tokenid']
        
        for i in data_feed.keys():
            before_dtype = data_feed[i].dtype
            if i in d_keys_int32:
                data_feed[i] = self.tensor2cuda(data_feed[i],INT)
            elif i in d_keys_int64:
                data_feed[i] = self.tensor2cuda(data_feed[i],LONG)
            else:
                data_feed[i] = self.tensor2cuda(data_feed[i],FLOAT)

        curr_ctx_feature = data_feed.curr_ctx_feature
        curr_img = data_feed.curr_img

        #retrieval
        text_in = curr_ctx_feature[:, 0]
        text_out = self.text_mlp(text_in)  # bs*2, dim_text_out     768 -> 768
        img_out = self.img_mlp(curr_img)  # bs*2, dim_text_out      2048 -> 128
        case_in = th.cat([text_out, img_out], dim=1)
        test_case_out = self.multi_mlp(case_in) # 896 -> 896 

        train_vecs_in = self.np2var(train_vecs, FLOAT)  # train_size, dim_case_out  
        cos_sim = self.sim(test_case_out.unsqueeze(1), train_vecs_in.unsqueeze(0))  # bs, train_size   计算与train数据的余弦相似度

        maxk = max(topk+1)
        topk_values, topk_caseids_out = cos_sim.topk(maxk, 1, True, True)  # bs, K  根据余弦相似度来检索前k个最相似的值和相应的案例ID
        candidate_scores, candidate_caseids = None, None
        if num_cluster_candidates:
            candidate_scores, candidate_caseids = cos_sim.topk(num_cluster_candidates+1, 1, True, True)  # bs, K  

        return test_case_out, topk_caseids_out, candidate_scores, candidate_caseids
    
    def get_rsp(self, data_feed):  # get similar response weights 
        
        batch_size =data_feed['batch_size']
        d_keys_float32 = [ 'curr_ctx_feature', 'curr_rsp_feature', 'curr_img', 'curr_rsps_teach_tokenid', 'curr_rsps_target_tokenid', 'sim_ctx_feature', 'sim_rsp_feature', 'sim_imgs']
        d_keys_int32 = ['topk_caseids_in','caseid', 'curr_ctx_tokenmask',  'curr_rsp_tokenmask',  'sim_ctx_tokenmask', 'sim_rsp_tokenmask', 'sim_ids', 'batch_size']
        d_keys_int64 = ['curr_rsp_tokenid', 'curr_ctx_tokenid', 'curr_rsps_teach_tokenid', 'curr_rsps_target_tokenid', 'sim_rsp_tokenid', 'sim_ctx_tokenid']
        
        for i in data_feed.keys():
            before_dtype = data_feed[i].dtype
            if i in d_keys_int32:
                data_feed[i] = self.tensor2cuda(data_feed[i],INT)
            elif i in d_keys_int64:
                data_feed[i] = self.tensor2cuda(data_feed[i],LONG)
            else:
                data_feed[i] = self.tensor2cuda(data_feed[i],FLOAT)
                
        num_sim = self.num_sim
        batch_size = data_feed['batch_size']
        
        curr_ctx_feature = data_feed['curr_ctx_feature']
        curr_ctx_tokenid = data_feed['curr_ctx_tokenid']
        curr_ctx_tokenmask = data_feed['curr_ctx_tokenmask']
        curr_rsp_feature = data_feed['curr_rsp_feature']
        curr_rsp_tokenid = data_feed['curr_rsp_tokenid']
        curr_rsp_tokenmask = data_feed['curr_rsp_tokenmask']

        sim_ctx_feature = data_feed['sim_ctx_feature']
        sim_ctx_tokenid = data_feed['sim_ctx_tokenid']
        sim_ctx_tokenmask = data_feed['sim_ctx_tokenmask']
        sim_rsp_feature = data_feed['sim_rsp_feature']
        sim_rsp_tokenid = data_feed['sim_rsp_tokenid']
        sim_rsp_tokenmask = data_feed['sim_rsp_tokenmask']
        
    
        sim_ids = data_feed['sim_ids']
#         sim_imgs = data_feed['sim_imgs']

        #reuse
        ctx_embedded, ctx_enc_hidden, ctx_enc_pad_mask = self.ctx_encoder(curr_ctx_feature, curr_ctx_tokenmask) 
        sim_ctx_embedded, sim_ctx_enc_hidden, sim_ctx_enc_pad_mask = self.ctx_encoder(sim_ctx_feature, sim_ctx_tokenmask) 


        # compute sim rsp copy weights
        k_ctx_embedded = ctx_embedded.unsqueeze(1).repeat(1, num_sim, 1).reshape(batch_size * num_sim,2 * self.hidden_dim)
        scores = self.sim_score(k_ctx_embedded, sim_ctx_embedded)
        scores = scores.reshape(batch_size, num_sim)
        scores = F.softmax(scores, dim=-1)  # b, k

        rsp_embedded, rsp_enc_hidden, rsp_enc_pad_mask = self.rsp_encoder(curr_rsp_feature, curr_rsp_tokenmask) # b * 2h, b * l * 2h
        
        # rsp_embedded 为当前rsp的编码表征
        # scores 为当前ctx与simctx的相似度
        return rsp_embedded, scores

    

