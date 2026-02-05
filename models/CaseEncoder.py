import os
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import json
from torch.autograd import Variable
from transformers import AutoModel, AutoTokenizer
from pprg_end2end.utils import INT, FLOAT, LONG, cast_type, summary
from pprg_end2end.case_util  import EOS, PAD, BOS
from pprg_end2end.models import MLPLayer, TripletLoss
from pprg_end2end.enc2dec import CtxEncoder, RspEncoder, RspDecoder
from pprg_end2end.criterions import Loss, NLLEntropy
from pprg_end2end.models import BaseModel
from pprg_end2end.utils import Pack, DATA_PATH
import ipdb
import logging
logger = logging.getLogger()


class PPRG(BaseModel):
#     def __init__(self,corpus, config):
#         super().__init__()
    def __init__(self, corpus, config):
        super(PPRG, self).__init__(config)
        self.module_retrieval = CaseEncoder(config)
        self.module_reuse = CCN_simctx_enc_fix(corpus, config)
        self.vocab = corpus.vocab
        
    def forward(self, data_feed):
        
        loss_result = {}        
        triplet_loss, curr_ctxs_feature, curr_ctxs_in,sim_ctxs_feature, sim_ctxs_in , sim_rsps_feature, sim_rsps_in= self.module_retrieval(data_feed)

        nll_loss = self.module_reuse(data_feed,curr_ctxs_feature, curr_ctxs_in, 
            sim_ctxs_feature, sim_ctxs_in ,

            sim_rsps_feature, sim_rsps_in)   
        
        loss_result['loss_triponline'] = triplet_loss
        loss_result['nll_loss'] = nll_loss
        
        return loss_result
    
    def genvec(self, data_feed):
    
        curr_ctxs_feature, curr_ctxs_in, curr_rsps_feature, curr_rsps_in= self.module_retrieval.genvec(data_feed)        
        return curr_ctxs_feature, curr_ctxs_in, curr_rsps_feature, curr_rsps_in
    
    def inference(self, data_feed):

        loss_result = {}        
        curr_ctxs_feature, curr_ctxs_in,sim_ctxs_feature, sim_ctxs_in , sim_rsps_feature, sim_rsps_in = self.module_retrieval.inference(data_feed)

        nll_loss, all_pred_words, curr_rsps_target= self.module_reuse.inference(
            data_feed, curr_ctxs_feature, curr_ctxs_in, 
            sim_ctxs_feature, sim_ctxs_in ,
            
            sim_rsps_feature, sim_rsps_in)        

        loss_result['nll_loss'] = nll_loss        

        return loss_result, all_pred_words, curr_rsps_target      
        
    
class CaseEncoder(BaseModel):
    def __init__(self, config):
        super(CaseEncoder, self).__init__(config)
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
        
        vocab_file = DATA_PATH + '/vocab.json'
        with open(vocab_file, "r") as f:
            self.vocab_dict = json.load(f)
            self.vocab = list(self.vocab_dict.keys())
        self.unk_id = self.vocab_dict['<unk>']
        
        self.init_net()
    def sent2id(self, sent):
        return [self.vocab_dict.get(t, self.unk_id) for t in sent]
    
    def init_net(self):
#         ipdb.set_trace()
        self.text_model = AutoModel.from_pretrained(self.text_model_name_or_path).to(self.device)   
#         self.tokenizer = AutoTokenizer.from_pretrained(self.text_model_name_or_path)
        self.text_mlp = MLPLayer(self.dim_text_in, self.dim_text_out)
        self.img_mlp = MLPLayer(self.dim_img_in, self.dim_img_out)
        self.sim = nn.CosineSimilarity(dim=-1)
        self.loss_fct = TripletLoss(margin=self.margin)
        
#             text_pair_in = text_pair_in, # list of str, batch_size * 2
#             img_pair_in = img_pair_in,  # batch_size * 2, dim_img_in
#             labels = labels,  # batch_size * 2
            

#             curr_ctxs_id = vec_curr_ctx_id,
#             curr_rsps = vec_curr_rsps, # bs, max_out_len
#             curr_rsps_teach =vec_curr_rsps_teach,
#             curr_rsps_target=vec_curr_rsps_target,
#             sim_ctxs_id = vec_sim_ctxs_id, # bs*k, max_sim_ctx_len
#             sim_rsps = vec_sim_rsps,
            
#             batch_size =batch_size



    def forward(self, data_feed):
#         ipdb.set_trace()
        batch_size =data_feed['batch_size']
        
#         text_pair_in = self.np2var(data_feed['text_pair_in'], FLOAT)  # batch_size * 2  当前ctx与相似样本的ctx
        img_pair_in = self.np2var(data_feed['img_pair_in'], FLOAT)  # batch_size * 2, dim_img_in
#         sim_ctx_in = self.np2var(data_feed['sim_ctxs'], FLOAT) 
        labels = self.np2var(data_feed['labels'], INT)  # batch_size * 2
#         ipdb.set_trace()
        
        #text_pair str2vec
        text_pair_in = {}
        text_pair_in['input_ids'] = self.np2var(data_feed['text_pair_in'], LONG)
        text_pair_in['token_type_ids'] = th.zeros(text_pair_in['input_ids'].shape, dtype=th.int32).to(self.device)
        text_pair_in['attention_mask'] = th.sign(text_pair_in['input_ids'] )     
        text_pair_in_vec = self.text_model(**text_pair_in, output_hidden_states=True, return_dict=True)
        #↑  Bert_keys(['last_hidden_state', 'pooler_output', 'hidden_states'])
        text_pair_last_hidden = text_pair_in_vec.last_hidden_state#16 182 768
        text_pair_pooler_output = text_pair_in_vec.pooler_output#16 768
        
        #curr_ctx str2vec
        curr_ctxs_in = {}
        curr_ctxs_in['input_ids'] = self.np2var(data_feed['curr_ctxs_id'], LONG)
        curr_ctxs_in['token_type_ids'] = th.zeros(curr_ctxs_in['input_ids'].shape, dtype=th.int32).to(self.device)
        curr_ctxs_in['attention_mask'] = th.sign(curr_ctxs_in['input_ids'] )
        curr_ctxs_in_vec = self.text_model(**curr_ctxs_in, output_hidden_states=True, return_dict=True)
        curr_ctxs_last_hidden = curr_ctxs_in_vec.last_hidden_state#16 182 768
        curr_ctxs_pooler_output = curr_ctxs_in_vec.pooler_output#16 768
#         ipdb.set_trace()
        
        #sim_ctx str2vec
        sim_ctxs_in = {}  
        sim_ctxs_in['input_ids'] = self.np2var(data_feed['sim_ctxs_id'], LONG)
        sim_ctxs_in['token_type_ids'] = th.zeros(sim_ctxs_in['input_ids'].shape, dtype=th.int32).to(self.device)
        sim_ctxs_in['attention_mask'] = th.sign(sim_ctxs_in['input_ids'] )
        sim_ctxs_in_vec = self.text_model(**sim_ctxs_in, output_hidden_states=True, return_dict=True)
        sim_ctxs_last_hidden = sim_ctxs_in_vec.last_hidden_state#16 182 768
        sim_ctxs_pooler_output = sim_ctxs_in_vec.pooler_output#16 768        
        
        #curr_rsp str2vec
#         curr_rsps_teach_in = {}  
#         curr_rsps_teach_in['input_ids'] = self.np2var(data_feed['curr_rsps_teach'], LONG)
#         curr_rsps_teach_in['token_type_ids'] = th.zeros(curr_rsps_teach_in['input_ids'].shape, dtype=th.int32).to(self.device)
#         curr_rsps_teach_in['attention_mask'] = th.sign(curr_rsps_teach_in['input_ids'] )
#         curr_rsps_teach_in_vec = self.text_model(**curr_rsps_teach_in, output_hidden_states=True, return_dict=True)
#         curr_rsps_teach_last_hidden = curr_rsps_teach_in_vec.last_hidden_state#16 182 768
#         curr_rsps_teach_pooler_output = curr_rsps_teach_in_vec.pooler_output#16 768  
        
        #sim_rsp str2vec
        sim_rsps_in = {}  
        sim_rsps_in['input_ids'] = self.np2var(data_feed['sim_rsps'], LONG)
        sim_rsps_in['token_type_ids'] = th.zeros(sim_rsps_in['input_ids'].shape, dtype=th.int32).to(self.device)
        sim_rsps_in['attention_mask'] = th.sign(sim_rsps_in['input_ids'] )
        sim_rsps_in_vec = self.text_model(**sim_rsps_in, output_hidden_states=True, return_dict=True)
        sim_rsps_last_hidden = sim_rsps_in_vec.last_hidden_state#16 182 768
        sim_rsps_pooler_output = sim_rsps_in_vec.pooler_output#16 768  

        text_pair_mid = None
        if self.text_pooler == 'cls':
            # There is a linear+activation layer after CLS representation
            text_pair_mid = text_pair_pooler_output
        elif self.text_pooler == 'cls_before_pooler':
            text_pair_mid = text_pair_last_hidden[:, 0]
#         ipdb.set_trace()

        text_pair_out = self.text_mlp(text_pair_mid)  # bs*2, dim_text_out     768 -> 768
        img_pair_out = self.img_mlp(img_pair_in)  # bs*2, dim_text_out      2048 -> 128
        case_out = th.cat([text_pair_out, img_pair_out], dim=1)  # 896 -> 896

        loss = self.loss_fct(case_out, labels)

        return loss, curr_ctxs_last_hidden, curr_ctxs_in, sim_ctxs_last_hidden, sim_ctxs_in,  sim_rsps_last_hidden, sim_rsps_in
#         return loss_dict, text_pair_out[:batch_size,:], data_feed['text_rsp']

    def genvec(self, data_feed):
#         ipdb.set_trace()
        batch_size =data_feed['batch_size']

        
        #curr_ctx str2vec
        curr_ctxs_in = {}
        curr_ctxs_in['input_ids'] = self.np2var(data_feed['curr_ctxs_id'], LONG)
        curr_ctxs_in['token_type_ids'] = th.zeros(curr_ctxs_in['input_ids'].shape, dtype=th.int32).to(self.device)
        curr_ctxs_in['attention_mask'] = th.sign(curr_ctxs_in['input_ids'] )
        curr_ctxs_in_vec = self.text_model(**curr_ctxs_in, output_hidden_states=True, return_dict=True)
        curr_ctxs_last_hidden = curr_ctxs_in_vec.last_hidden_state#16 182 768
       
     

        curr_rsps_in = {}  
        curr_rsps_in['input_ids'] = self.np2var(data_feed['curr_rsps'], LONG)
        curr_rsps_in['token_type_ids'] = th.zeros(curr_rsps_in['input_ids'].shape, dtype=th.int32).to(self.device)
        curr_rsps_in['attention_mask'] = th.sign(curr_rsps_in['input_ids'] )
        curr_rsps_in_vec = self.text_model(**curr_rsps_in, output_hidden_states=True, return_dict=True)
        curr_rsps_last_hidden = curr_rsps_in_vec.last_hidden_state#16 182 768



        return curr_ctxs_last_hidden, curr_ctxs_in, curr_rsps_last_hidden, curr_rsps_in
     
    def inference(self, data_feed):
#         ipdb.set_trace()
        batch_size =data_feed['batch_size']       

        labels = self.np2var(data_feed['labels'], INT)  # batch_size * 2
            
        #curr_ctx str2vec
        with torch.no_grad():
            curr_ctxs_in = {}
            curr_ctxs_in['input_ids'] = self.np2var(data_feed['curr_ctxs_id'], LONG)
            curr_ctxs_in['token_type_ids'] = th.zeros(curr_ctxs_in['input_ids'].shape, dtype=th.int32).to(self.device)
            curr_ctxs_in['attention_mask'] = th.sign(curr_ctxs_in['input_ids'] )
            curr_ctxs_in_vec = self.text_model(**curr_ctxs_in, output_hidden_states=True, return_dict=True)
            curr_ctxs_last_hidden = curr_ctxs_in_vec.last_hidden_state#16 182 768
            curr_ctxs_pooler_output = curr_ctxs_in_vec.pooler_output#16 768
    #         ipdb.set_trace()

            #sim_ctx str2vec
            sim_ctxs_in = {}  
            sim_ctxs_in['input_ids'] = self.np2var(data_feed['sim_ctxs_id'], LONG)
            sim_ctxs_in['token_type_ids'] = th.zeros(sim_ctxs_in['input_ids'].shape, dtype=th.int32).to(self.device)
            sim_ctxs_in['attention_mask'] = th.sign(sim_ctxs_in['input_ids'] )
            sim_ctxs_in_vec = self.text_model(**sim_ctxs_in, output_hidden_states=True, return_dict=True)
            sim_ctxs_last_hidden = sim_ctxs_in_vec.last_hidden_state#16 182 768
            sim_ctxs_pooler_output = sim_ctxs_in_vec.pooler_output#16 768    

                    #curr_rsp str2vec
    #         curr_rsps_teach_in = {}  
    #         curr_rsps_teach_in['input_ids'] = self.np2var(data_feed['curr_rsps_teach'], LONG)
    #         curr_rsps_teach_in['token_type_ids'] = th.zeros(curr_rsps_teach_in['input_ids'].shape, dtype=th.int32).to(self.device)
    #         curr_rsps_teach_in['attention_mask'] = th.sign(curr_rsps_teach_in['input_ids'] )
    #         curr_rsps_teach_in_vec = self.text_model(**curr_rsps_teach_in, output_hidden_states=True, return_dict=True)
    #         curr_rsps_teach_last_hidden = curr_rsps_teach_in_vec.last_hidden_state#16 182 768
    #         curr_rsps_teach_pooler_output = curr_rsps_teach_in_vec.pooler_output#16 768  

            #sim_rsp str2vec
            sim_rsps_in = {}  
            sim_rsps_in['input_ids'] = self.np2var(data_feed['sim_rsps'], LONG)
            sim_rsps_in['token_type_ids'] = th.zeros(sim_rsps_in['input_ids'].shape, dtype=th.int32).to(self.device)
            sim_rsps_in['attention_mask'] = th.sign(sim_rsps_in['input_ids'] )
            sim_rsps_in_vec = self.text_model(**sim_rsps_in, output_hidden_states=True, return_dict=True)
            sim_rsps_last_hidden = sim_rsps_in_vec.last_hidden_state#16 182 768
            sim_rsps_pooler_output = sim_rsps_in_vec.pooler_output#16 768 


        return curr_ctxs_last_hidden, curr_ctxs_in, sim_ctxs_last_hidden, sim_ctxs_in, sim_rsps_last_hidden, sim_rsps_in
    
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
        ipdb.set_trace()
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
        img_out = self.img_mlp(img_in)    # bs, dim_text_out
        test_case_out = th.cat([text_out, img_out], dim=1)

        train_vecs_in = self.np2var(train_vecs, FLOAT)  # train_size, dim_case_out  
        cos_sim = self.sim(test_case_out.unsqueeze(1), train_vecs_in.unsqueeze(0))  # bs, train_size   计算与train数据的余弦相似度

        maxk = max(topk)
        topk_values, topk_caseids_out = cos_sim.topk(maxk, 1, True, True)  # bs, K  根据余弦相似度来检索前k个最相似的值和相应的案例ID

        candidate_scores, candidate_caseids = None, None
        if num_cluster_candidates:
            candidate_scores, candidate_caseids = cos_sim.topk(num_cluster_candidates, 1, True, True)  # bs, K  

        return test_case_out, topk_caseids_out, candidate_scores, candidate_caseids
    
    
    
class CCN_simctx_enc_fix(BaseModel):
    def __init__(self, corpus, config):
        super(CCN_simctx_enc_fix, self).__init__(config)
#     def __init__(self, config):
#         super().__init__()
        self.vocab = corpus.vocab
        self.vocab_dict = corpus.vocab_dict
        self.vocab_size = len(self.vocab)
        self.bos_id = self.vocab_dict[BOS]
        self.eos_id = self.vocab_dict[EOS]
        self.pad_id = self.vocab_dict[PAD]
        self.embedding_dim = config.embed_size
        self.hidden_dim = self.config.dec_cell_size
        self.num_sim = self.config.num_sim
        self.init_net()

    def init_net(self):
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
        # self.nll = NLLEntropy(self.pad_id)

    def valid_loss(self, losses, batch_cnt=None, add_loss_adv=None):
        # to (weighted) sum up losses
        total_loss = 0
        for key, loss in losses.items():
            total_loss += loss
        return total_loss

    def forward(self, data_feed, curr_ctxs_feature, curr_ctxs_in, sim_ctxs_feature, sim_ctxs_in , sim_rsps_feature, sim_rsps_in):
  #curr_ctx_feature, curr_ctx_id, sim_ctx_feature, sim_ctx_id

#         curr_rsps = self.np2var(data_feed['curr_rsps'], LONG) # bs, max_out_len
        curr_rsps_teach = self.np2var(data_feed['curr_rsps_teach'], LONG)
        curr_rsps_target = self.np2var(data_feed['curr_rsps_target'], LONG)
#         sim_ctxs = self.np2var(data_feed['sim_ctxs'], LONG) # bs*k, max_sim_ctx_len
        sim_rsps = self.np2var(data_feed['sim_rsps'], LONG)
        num_sim = self.num_sim
        batch_size = data_feed['batch_size']
#         ipdb.set_trace()

        # result to store loss
        loss_result = {}
        # encoding   dim=600
        ctx_embedded, ctx_enc_hidden, ctx_enc_pad_mask = self.ctx_encoder(curr_ctxs_feature, curr_ctxs_in['attention_mask']) # [B, 2H], [B, L, 2H]
        sim_ctx_embedded, sim_ctx_enc_hidden, sim_ctx_enc_pad_mask = self.ctx_encoder(sim_ctxs_feature, sim_ctxs_in['attention_mask'])  # [B*K, 2H], [B*K, L, 2H]

        
        sim_rsp_embedded, sim_rsp_enc_hidden, sim_rsp_enc_pad_mask = self.rsp_encoder(sim_rsps_feature, sim_rsps_in['attention_mask'])  # [B*K, 2H], [B*K, L', 2H]

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
        dec_emb = self.embedding(curr_rsps_teach)  # [B x T x E]
#         dec_emb = curr_rsps_teach_feature # [B x T x E]

        hidden = ctx_embedded[:, :self.hidden_dim] + ctx_embedded[:, self.hidden_dim:] # [B, H]
        hidden = hidden.unsqueeze(0) # [1, B, H]
        k_hidden = hidden.squeeze(0).unsqueeze(1).repeat(1, num_sim, 1).reshape(batch_size * num_sim,self.hidden_dim).unsqueeze(0)#初始解码器状态,信息来自curr_rsps_teach

        for t in range(self.config.max_dec_len):
#             ipdb.set_trace() 
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
            vertical_dist = vocab_dist.scatter_add(dim=1, index=curr_ctxs_in['input_ids'], src=weighted_context_attn_dist)#??

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
            sim_rsps_stacked = sim_rsps.reshape(batch_size, num_sim, -1) # b, k, L'
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
            
            
            #
            _, pred_word = horizontal_dist.topk(1) # b,1
            all_pred_words.append(pred_word.clone())

            
        final_dists = th.stack(horizontal_dists, dim=-1)  # [B x V_x x T]   增加一维
        all_pred_words = th.cat(all_pred_words, dim=1).squeeze() # b, L'   32 50 预测的词id
        nll_loss = self.criterion.nll_loss(final_dists, curr_rsps_target)
#         ipdb.set_trace()
        return nll_loss

        
    def inference(self, data_feed, curr_ctxs_feature, curr_ctxs_in, sim_ctxs_feature, sim_ctxs_in ,sim_rsps_feature, sim_rsps_in):
        
#         curr_rsps = self.np2var(data_feed['curr_rsps'], LONG) # bs, max_out_len
        curr_rsps_teach = self.np2var(data_feed['curr_rsps_teach'], LONG)
        curr_rsps_target = self.np2var(data_feed['curr_rsps_target'], LONG)
        sim_rsps = self.np2var(data_feed['sim_rsps'], LONG)
        num_sim = self.num_sim
        batch_size =  data_feed['batch_size']

        # encoding
        ctx_embedded, ctx_enc_hidden, ctx_enc_pad_mask = self.ctx_encoder(curr_ctxs_feature, curr_ctxs_in['attention_mask']) # [B, 2H], [B, L, 2H]
        sim_ctx_embedded, sim_ctx_enc_hidden, sim_ctx_enc_pad_mask = self.ctx_encoder(sim_ctxs_feature, sim_ctxs_in['attention_mask'])  # [B*K, 2H], [B*K, L, 2H]
        sim_rsp_embedded, sim_rsp_enc_hidden, sim_rsp_enc_pad_mask = self.rsp_encoder(sim_rsps_feature, sim_rsps_in['attention_mask'])  # [B*K, 2H], [B*K, L', 2H]  # [B*K, 2H], [B*K, L', 2H]

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
        dec_emb = self.embedding(curr_rsps_teach)
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
            vertical_dist = vocab_dist.scatter_add(dim=-1, index=curr_ctxs_in['input_ids'], src=weighted_context_attn_dist)

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
            sim_rsps_stacked = sim_rsps.reshape(batch_size, num_sim, -1) # b, k, L'
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
        nll_loss = self.criterion.nll_loss(final_dists, curr_rsps_target)


        return nll_loss, all_pred_words, curr_rsps_target

    def get_rsp(self, data_feed):  # get similar response weights 
        curr_ctxs = self.np2var(data_feed['curr_ctxs'], LONG)  # bs, max_curr_ctx_len
        curr_rsps = self.np2var(data_feed['curr_rsps'], LONG)  # bs, max_out_len
        sim_ctxs = self.np2var(data_feed['sim_ctxs'], LONG)  # bs*k, max_sim_ctx_len
        sim_rsps = self.np2var(data_feed['sim_rsps'], LONG)
        num_sim = self.num_sim
        batch_size = len(curr_ctxs)

        # encoding
        ctx_embedded, ctx_enc_hidden, ctx_enc_pad_mask = self.ctx_encoder(curr_ctxs)  # [B, 2H], [B, L, 2H]
        sim_ctx_embedded, sim_ctx_enc_hidden, sim_ctx_enc_pad_mask = self.ctx_encoder(sim_ctxs)  # [B*K, 2H], [B*K, L, 2H]

        # compute sim rsp copy weights
        k_ctx_embedded = ctx_embedded.unsqueeze(1).repeat(1, num_sim, 1).reshape(batch_size * num_sim,2 * self.hidden_dim)
        scores = self.sim_score(k_ctx_embedded, sim_ctx_embedded)
        scores = scores.reshape(batch_size, num_sim)
        scores = F.softmax(scores, dim=-1)  # b, k

        rsp_embedded, rsp_enc_hidden, rsp_enc_pad_mask = self.rsp_encoder(curr_rsps) # b * 2h, b * l * 2h

        return rsp_embedded, scores


