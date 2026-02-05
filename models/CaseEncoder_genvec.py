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
from pprg_end2end.enc2dec import CtxEncoder, RspEncoder, RspDecoder,CtxEncoder_2, RspEncoder_2
from pprg_end2end.criterions import Loss, NLLEntropy
from pprg_end2end.models import BaseModel
from pprg_end2end.utils import Pack, DATA_PATH
import ipdb
import logging
logger = logging.getLogger()


class PPRG_genvec(BaseModel):
#     def __init__(self,corpus, config):
#         super().__init__()
    def __init__(self, corpus, config):
        super(PPRG_genvec, self).__init__(config)
        self.module_genvec = CaseEncoder(corpus, config)
        vocab_file = DATA_PATH + '/vocab_bert.json'
        with open(vocab_file, "r") as f:
            self.vocab_dict = json.load(f)
            self.vocab = list(self.vocab_dict.keys())
            self.vocab_size = len(self.vocab)
        
    def forward(self, data_feed):       
        loss_result = self.module_genvec(data_feed)
        
        return loss_result
    def genvec(self, data_feed):
    
        curr_ctxs_feature, curr_ctxs_in, curr_rsps_feature, curr_rsps_in= self.module_genvec.genvec(data_feed)        
        return curr_ctxs_feature, curr_ctxs_in, curr_rsps_feature, curr_rsps_in
    
    def inference(self, data_feed):
        loss_result, all_pred_words, curr_rsps_target= self.module_genvec.inference(data_feed)      

        return loss_result, all_pred_words, curr_rsps_target   
    
class CaseEncoder(BaseModel):
    def __init__(self,corpus,config):
        super(CaseEncoder, self).__init__(config)  
        self.config = config
        self.use_gpu = config.use_gpu
        self.device = th.device("cuda:0" if  self.use_gpu else "cpu")
        self.margin = config.margin
        self.text_model_name_or_path = config.text_model_name_or_path
        self.text_pooler = config.text_pooler  # 'cls' / 'cls_before_pooler'
        self.max_ctx_len = config.max_ctx_len
        self.max_rsp_len = config.max_dec_len
        
        self.dim_text_in = config.dim_text_in
        self.dim_text_out = config.dim_text_out
        self.dim_img_in = config.dim_img_in
        self.dim_img_out = config.dim_img_out
        self.dim_case_out = self.dim_text_out +  self.dim_img_out

        self.init_net()

    def init_net(self):
        self.text_model = AutoModel.from_pretrained(self.text_model_name_or_path).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(self.text_model_name_or_path)
    
    def genvec(self, data_feed):
        
        curr_ctx_in_id = self.tokenizer(data_feed['curr_ctx'], return_tensors='pt', padding='max_length',
                                      max_length=self.max_ctx_len, truncation=True).to(self.device)#16 182
        curr_ctx_in_vec = self.text_model(**curr_ctx_in_id, output_hidden_states=True, return_dict=True)
        #Bert  odict_keys(['last_hidden_state', 'pooler_output', 'hidden_states'])
        curr_ctx_last_hidden = curr_ctx_in_vec.last_hidden_state#16 182 768
#         ipdb.set_trace()
        curr_rsp_in_id = self.tokenizer(data_feed['curr_rsp'], return_tensors='pt', padding='max_length', max_length=self.max_rsp_len, truncation=True).to(self.device)#16 182
        curr_rsp_in_vec = self.text_model(**curr_rsp_in_id, output_hidden_states=True, return_dict=True)
        curr_rsp_last_hidden = curr_rsp_in_vec.last_hidden_state#16 182 768        
        
        
        
        return curr_ctx_last_hidden, curr_ctx_in_id, curr_rsp_last_hidden, curr_rsp_in_id
    
#     def genvec(self, data_feed):
# #         ipdb.set_trace()

        
#         #curr_ctx str2vec
#         curr_ctxs_in = {}
#         curr_ctxs_in['input_ids'] = self.np2var(data_feed['curr_ctxs_id'], LONG)
#         curr_ctxs_in['token_type_ids'] = th.zeros(curr_ctxs_in['input_ids'].shape, dtype=th.int32).to(self.device)
#         curr_ctxs_in['attention_mask'] = th.sign(curr_ctxs_in['input_ids'] )
#         curr_ctxs_in_vec = self.text_model(**curr_ctxs_in, output_hidden_states=True, return_dict=True)
#         curr_ctxs_last_hidden = curr_ctxs_in_vec.last_hidden_state#16 182 768
       
     

#         curr_rsps_in = {}  
#         curr_rsps_in['input_ids'] = self.np2var(data_feed['curr_rsps'], LONG)
#         curr_rsps_in['token_type_ids'] = th.zeros(curr_rsps_in['input_ids'].shape, dtype=th.int32).to(self.device)
#         curr_rsps_in['attention_mask'] = th.sign(curr_rsps_in['input_ids'] )
#         curr_rsps_in_vec = self.text_model(**curr_rsps_in, output_hidden_states=True, return_dict=True)
#         curr_rsps_last_hidden = curr_rsps_in_vec.last_hidden_state#16 182 768



#         return curr_ctxs_last_hidden, curr_ctxs_in, curr_rsps_last_hidden, curr_rsps_in
    
