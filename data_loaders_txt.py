import logging
import numpy as np

from pprg_end2end.utils import Pack
import ipdb
logger = logging.getLogger()

class BaseDataLoader(object):
    def __init__(self, mode):
        self.data_size = None
        self.indexes = None
        self.mode = mode

    def _shuffle_indexes(self):
        np.random.shuffle(self.indexes)

    def epoch_init(self, config, shuffle=True, verbose=True, drop_last=True):
        self.ptr = 0
        self.batch_size = config.batch_size
        if drop_last:
            self.num_batch = self.data_size // config.batch_size
        else:
            sized = self.data_size % config.batch_size == 0
            num_batch = self.data_size // config.batch_size
            self.num_batch = num_batch if sized else (num_batch + 1)

        if verbose:
            if drop_last:
                print('Number of left over sample = %d' % (self.data_size - config.batch_size * self.num_batch))
            else:
                print('Do not drop last sample')

        if shuffle:
            self._shuffle_indexes()

        self.batch_indexes = []
        for i in range(self.num_batch):
            self.batch_indexes.append(self.indexes[i*self.batch_size: min((i+1)*self.batch_size, self.data_size)])

        if verbose:
            print('%s begins with %d batches' % (self.mode, self.num_batch))

    def next_batch(self):
        if self.ptr < self.num_batch:
            selected_ids = self.batch_indexes[self.ptr]
            self.ptr += 1
            return self._prepare_batch(selected_index=selected_ids)
        else:
            return None

    def _prepare_batch(self, *args, **kwargs):
        raise NotImplementedError('Have to override _prepare_batch()')
        
    def pad_to(self, max_len, tokens, do_pad):#多的裁掉，少的补0
#         ipdb.set_trace()
        if len(tokens) >= max_len:
            return tokens[: max_len-1] + [tokens[-1]]
        elif do_pad:
            return tokens + [0] * (max_len - len(tokens))
        else:
            return tokens


class CaseDataLoader(BaseDataLoader):

    def __init__(self, mode, data, config):
        super(CaseDataLoader, self).__init__(mode)
        self.mode = mode  # 'train'/'val'/'test'
        self.config = config
        self.numpos = config.numpos if config.numpos else config.K
        self.data = data  # case-level
        self.data_size = len(self.data)  # number of cases
        self.indexes = [id for id in range(self.data_size)]
        # build positive paired data
        self.data_pairs = self.build_pairs()  # pair-level 
        self.data_size_pairs = len(self.data_pairs)  # number of positive pairs
        self.indexes_pairs = [id for id in range(self.data_size_pairs)]
        
        self.num_sim = self.config.num_sim
        self.max_curr_ctx_len = self.config.max_ctx_len
     
    def build_pairs(self):
#         ipdb.set_trace()
        results = []
        for caseinfo in self.data:
            for k, caseid_p in enumerate(caseinfo.sim_ids):#相似样本
                if k >= self.numpos:
                    break
                pos_ctx = caseinfo.sim_ctxs[k]
                pos_ctx_tokenid = caseinfo.sim_ctxs_tokenid[k]
                pos_caseid = caseinfo.sim_ids[k]
                pos_img = caseinfo.sim_imgs[k]
#                 ipdb.set_trace()
                currcase = Pack(
                    caseid = caseinfo.caseid,  # int                    
                    curr_ctx = caseinfo.curr_ctx,
                    curr_ctx_tokenid = caseinfo.curr_ctx_tokenid,
                    curr_rsp = caseinfo.curr_rsp,
                    curr_rsp_tokenid = caseinfo.curr_rsp_tokenid,
                    curr_img = caseinfo.curr_img,
                    
                    pos_ctx = pos_ctx,
                    pos_ctx_tokenid = pos_ctx_tokenid,
                    pos_caseid = pos_caseid,
                    pos_img = pos_img,
                    
                    sim_ctx = caseinfo.sim_ctxs,
                    sim_ctx_tokenid = caseinfo.sim_ctxs_tokenid,
                    sim_rsp = caseinfo.sim_rsps,
                    sim_rsp_tokenid = caseinfo.sim_rsps_tokenid,
                    sim_ids = caseinfo.sim_ids,
                    sim_imgs = caseinfo.sim_imgs
                )
                results.append(currcase)
        return results

    
    # case level
    def epoch_init(self, config, shuffle=True, verbose=True, drop_last=True):
        self.ptr = 0
        self.batch_size = config.batch_size
        if drop_last:
            self.num_batch = self.data_size // config.batch_size
        else:
            sized = self.data_size % config.batch_size == 0
            num_batch = self.data_size // config.batch_size
            self.num_batch = num_batch if sized else (num_batch + 1)

        if shuffle:
            self._shuffle_indexes()

        self.batch_indexes = []  # updated by batch_size
        for i in range(self.num_batch):
            self.batch_indexes.append(self.indexes[i * self.batch_size: min((i+1)*self.batch_size, self.data_size)])

        if verbose:
            if drop_last:
                print('Number of left over sample = %d' % (self.data_size - config.batch_size * self.num_batch))
            else:
                print('Do not drop last sample')

        if verbose:
            print('%s begins with %d batches' % (self.mode, self.num_batch))

    def _prepare_batch(self, selected_index):
   

        cases = [self.data[idx] for idx in selected_index]
        batch_size = len(selected_index)#32
        dim_img_in = cases[0].imgvec.size
        K = self.config.K       
       
        max_dec_len = self.config.max_dec_len#50
        max_curr_ctx_len = self.max_curr_ctx_len
        max_sim_ctx_len = self.max_curr_ctx_len
        
#         for batch_id, case in enumerate(cases):
#             max_curr_ctx_len = max_curr_ctx_len if len(case.curr_ctx_id) < max_curr_ctx_len else len(case.curr_ctx_id)
#             max_sim_ctx_len = max_sim_ctx_len if case.max_sim_ctx_len < max_sim_ctx_len else case.max_sim_ctx_len
#         max_curr_ctx_len = max_curr_ctx_len if max_curr_ctx_len < self.config.max_ctx_len else self.config.max_ctx_len
#         max_sim_ctx_len = max_sim_ctx_len if max_sim_ctx_len < self.config.max_ctx_len else self.config.max_ctx_len
        

        caseids = []

        img_in = np.zeros((batch_size, dim_img_in), dtype=float)
        topk_caseids_in = np.full((batch_size, K), -1, dtype=int)
        
        curr_ctx_id, curr_rsps, curr_rsps_teach, curr_rsps_target = [], [], [], []
        sim_ctxs_id, sim_rsps = [], []
        for batch_id, case in enumerate(cases):
            caseids.append(case.caseid)

            topk_caseids = np.array(case.topk_caseids, dtype=int)
            topk_caseids_in[batch_id, :len(case.topk_caseids)] = topk_caseids
            
            
            curr_ctx_id.append([t for _, t in enumerate(self.pad_to(max_curr_ctx_len, case.curr_ctx_id, do_pad=True))])
            curr_rsps.append([t for _, t in enumerate(self.pad_to(max_dec_len, case.curr_rsp, do_pad=True))])
            curr_rsps_teach.append([t for _, t in enumerate(self.pad_to(max_dec_len, (case.curr_rsp)[:-1], do_pad=True))])#去掉了<eos>
            curr_rsps_target.append([t for _, t in enumerate(self.pad_to(max_dec_len, (case.curr_rsp)[1:], do_pad=True))])#去掉了<s>

            for k, sim_ctx_id in enumerate(case.sim_ctxs_id):                
                sim_ctxs_id.append([t for _, t in enumerate(self.pad_to(max_sim_ctx_len, sim_ctx_id, do_pad=True))])   
            for k, sim_rsp in enumerate(case.sim_rsps):
                sim_rsps.append([t for _, t in enumerate(self.pad_to(max_dec_len, sim_rsp, do_pad=True))])

        vec_curr_ctx_id = np.zeros((batch_size, max_curr_ctx_len), dtype=np.int32)    
        vec_curr_rsps = np.zeros((batch_size, max_dec_len), dtype=np.int32)
        vec_curr_rsps_teach = np.zeros((batch_size, max_dec_len), dtype=np.int32)
        vec_curr_rsps_target = np.zeros((batch_size, max_dec_len), dtype=np.int32)
        vec_sim_ctxs_id = np.zeros((batch_size * self.num_sim, max_sim_ctx_len), dtype=np.int32)
        vec_sim_rsps = np.zeros((batch_size * self.num_sim, max_dec_len), dtype=np.int32)
        
        labels = caseids + caseids#BS*2 
        labels = np.array(labels, dtype=int)

        for batch_id, case in enumerate(cases):#list to array


            img_in[batch_id, :] = case.imgvec  #array
         
            vec_curr_ctx_id[batch_id, :max_curr_ctx_len] = curr_ctx_id[batch_id]#32 123
            vec_curr_rsps[batch_id, :max_dec_len] = curr_rsps[batch_id]#32 50
            vec_curr_rsps_teach[batch_id, :max_dec_len] = curr_rsps_teach[batch_id]
            vec_curr_rsps_target[batch_id, :max_dec_len] = curr_rsps_target[batch_id]
            
            for k, sim_ctx in enumerate(case.sim_ctxs_id):
                vec_sim_ctxs_id[batch_id*self.num_sim+k] = sim_ctxs_id[batch_id*self.num_sim+k]
            for k, sim_rsp in enumerate(case.sim_rsps):
                vec_sim_rsps[batch_id*self.num_sim+k, :max_dec_len] = sim_rsps[batch_id*self.num_sim+k]

        return Pack(        
                                   
            
            img_in = img_in,#图片
            topk_caseids_in = topk_caseids_in, #相似样本ID
            labels = labels,
            
            curr_ctxs_id = vec_curr_ctx_id,
            curr_rsps = vec_curr_rsps, # bs, max_out_len
            curr_rsps_teach =vec_curr_rsps_teach,
            curr_rsps_target=vec_curr_rsps_target,
            sim_ctxs_id = vec_sim_ctxs_id, # bs*k, max_sim_ctx_len
            sim_rsps = vec_sim_rsps,
            
            batch_size =batch_size

        )
    
    

    # paired level
    def _shuffle_indexes_pairs(self):
        np.random.shuffle(self.indexes_pairs)

    def epoch_init_pairs(self, config, shuffle=True, verbose=True, drop_last=True):
        self.ptr = 0
        self.batch_size = config.batch_size
        if drop_last:
            self.num_batch_pairs = self.data_size_pairs // config.batch_size
        else:
            sized = self.data_size_pairs % config.batch_size == 0
            num_batch_pairs = self.data_size_pairs // config.batch_size
            self.num_batch_pairs = num_batch_pairs if sized else (num_batch_pairs + 1)

        if shuffle:
            self._shuffle_indexes_pairs()

        self.batch_indexes_pairs = []  # updated by batch_size
        for i in range(self.num_batch_pairs):
            self.batch_indexes_pairs.append(self.indexes_pairs[i * self.batch_size: min((i+1)*self.batch_size, self.data_size_pairs)])

        if verbose:
            if drop_last:
                print('Number of left over paired sample = %d' % (self.data_size_pairs - config.batch_size * self.num_batch_pairs))
            else:
                print('Do not drop last sample')

        if verbose:
            print('%s begins with %d batches' % (self.mode, self.num_batch_pairs))

    def next_batch_pairs(self):
        if self.ptr < self.num_batch_pairs:
            selected_ids = self.batch_indexes_pairs[self.ptr]
            self.ptr += 1
            return self._prepare_batch_pairs(selected_index=selected_ids)
        else:
            return None
#                     caseid = caseinfo.caseid,  # int                    
#                     curr_ctx = caseinfo.curr_ctx,
#                     curr_ctx_tokenid = caseinfo.curr_ctx_tokenid,
#                     curr_rsp = caseinfo.curr_rsp,
#                     curr_rsp_tokenid = caseinfo.curr_rsp_tokenid,
#                     curr_img = caseinfo.curr_img,
                    
#                     pos_ctx = pos_ctx,
#                     pos_ctx_tokenid = pos_ctx_tokenid,
#                     pos_caseid = pos_caseid,
#                     pos_img = pos_img,
                    
#                     sim_ctx = caseinfo.sim_ctx,
#                     sim_ctx_tokenid = caseinfo.sim_ctx_tokenid,
#                     sim_rsp = caseinfo.sim_rsp,
#                     sim_rsp_tokenid = caseinfo.sim_rsp_tokenid,
#                     sim_ids = caseinfo.sim_ids,
#                     sim_imgs = caseinfo.sim_imgs
    def _prepare_batch_pairs(self, selected_index):
#         ipdb.set_trace()
        cases = [self.data_pairs[idx] for idx in selected_index]    
    
        max_dec_len = self.config.max_dec_len#50
        max_curr_ctx_len = self.max_curr_ctx_len
        max_sim_ctx_len = self.max_curr_ctx_len
#     
        batch_size = len(selected_index)  # number of pairs in this batch
        dim_img_in = cases[0].curr_img.size  

        caseids = []
        
        curr_ctx, curr_ctx_tokenid = [],[]
        curr_rsp = []
        curr_rsp_teach_tokenid = []
        curr_rsp_target_tokenid = []
        sim_ctx, sim_ctx_tokenid = [],[]
        sim_rsp, sim_rsp_tokenid = [], []
        for batch_id, case in enumerate(cases):
#             ipdb.set_trace()
            caseids.append(case.caseid)
            curr_ctx.append(case.curr_ctx)  
            curr_rsp.append(case.curr_rsp)
            curr_ctx_tokenid.append([t for _, t in enumerate(self.pad_to(max_curr_ctx_len, case.curr_ctx_tokenid, do_pad=True))])
            curr_rsp_teach_tokenid.append([t for _, t in enumerate(self.pad_to(max_dec_len, (case.curr_rsp_tokenid)[:-1], do_pad=True))])#去掉了<eos>
            curr_rsp_target_tokenid.append([t for _, t in enumerate(self.pad_to(max_dec_len, (case.curr_rsp_tokenid)[1:], do_pad=True))])#去掉了<s>

            for k in range(len(case.sim_ids)): 
                sim_ctx.append(case.sim_ctx[k])
                sim_ctx_tokenid.append([t for _, t in enumerate(self.pad_to(max_sim_ctx_len, case.sim_ctx_tokenid[k], do_pad=True))])
            for k in range(len(case.sim_ids)):
                sim_rsp.append(case.sim_rsp[k])
                sim_rsp_tokenid.append([t for _, t in enumerate(self.pad_to(max_dec_len, case.sim_rsp_tokenid[k], do_pad=True))])



            

              
#         ipdb.set_trace()
        return Pack( 
            curr_ctx = curr_ctx,
            curr_ctx_tokenid = np.array(curr_ctx_tokenid),
            curr_rsp = curr_rsp,
            curr_rsp_teach_tokenid = np.array(curr_rsp_teach_tokenid),
            curr_rsp_target_tokenid = np.array(curr_rsp_target_tokenid),
            sim_ctx = sim_ctx,
            sim_ctx_tokenid = np.array(sim_ctx_tokenid),
            sim_rsp = sim_rsp,
            sim_rsp_tokenid =np.array(sim_rsp_tokenid)
            
            
        )

    def clone(self):
        return CaseDataLoader(self.mode, self.data, self.config)
    
    def pad_to(self, max_len, tokens, do_pad):
        if len(tokens) >= max_len:
            return tokens[: max_len-1] + [tokens[-1]]
        elif do_pad:
            return tokens + [0] * (max_len - len(tokens))
        else:
            return tokens

