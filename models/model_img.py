import numpy as np
import torch as th
import torch.nn as nn
from pprg_end2end.utils import FLOAT, LONG, cast_type
from pprg_end2end.models import BaseModel
from pprg_end2end.utils import Pack, DATA_PATH
import ipdb
import logging
logger = logging.getLogger()

class ImgRetriever(BaseModel):
    def __init__(self, config):
        super(ImgRetriever, self).__init__(config)
        self.config = config
        self.use_gpu = config.use_gpu
        self.margin = config.margin
        self.init_net()

    def init_net(self):
        self.sim = nn.CosineSimilarity(dim=-1)
        self.loss_fct = nn.TripletMarginLoss(margin=self.margin)

    def valid_loss(self, loss, batch_cnt=None, add_loss_adv=None):
        total_loss = 0.0
        for k, l in loss.items():
            if l is not None:
                total_loss += l
        return total_loss


    def forward(self, data_feed):
        weighted_sim_imgvec = self.np2var(data_feed['weighted_sim_imgvec_in'], FLOAT) # batch_size, dim_text_in
        gt_imgvec = self.np2var(data_feed['gt_imgvec_in'], FLOAT)  # batch_size, dim_text_in
        rd_imgvec = self.np2var(data_feed['rd_imgvec_in'], FLOAT)  # batch_size, dim_img_in

        anchor_imgvec = weighted_sim_imgvec

        loss = self.loss_fct(anchor_imgvec, gt_imgvec, rd_imgvec)
        loss_dict = {'loss_tripoffline': loss}
        return loss_dict

    def get_similarity(self, data_feed, train_vecs):
#         ipdb.set_trace()
        weighted_sim_imgvec = self.np2var(data_feed['weighted_sim_imgvec_in'], FLOAT)  # batch_size, dim_text_in
        anchor_imgvec = weighted_sim_imgvec
#         ipdb.set_trace()
        train_vecs_in = self.np2var(train_vecs, FLOAT) # train_size, dim_case_out
        cos_sim = self.sim(anchor_imgvec.unsqueeze(1), train_vecs_in.unsqueeze(0)) # bs, train_size
        
        maxk = 5
        topk_values, topk_caseids_out = cos_sim.topk(maxk, 1, True, True) # bs, K

        return topk_caseids_out