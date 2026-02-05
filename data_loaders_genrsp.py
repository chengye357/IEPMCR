import logging
import numpy as np
import torch
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
        self.indexes_pairs_shuffled = [id for id in range(self.data_size_pairs)]
        self.indexes_pairs_unshuffled = [id for id in range(self.data_size_pairs)]
        self.num_sim = self.config.num_sim
        self.max_curr_ctx_len = self.config.max_ctx_len

                    
    def build_pairs(self):
#         ipdb.set_trace()
        results = []
        for caseinfo in self.data:
            for k, pos_caseid in enumerate(caseinfo.sim_ids):#相似样本
                if k >= self.numpos:
                    break
                pos_ctx_feature = caseinfo.sim_ctx_feature[k]
                pos_ctx_tokenid = caseinfo.sim_ctx_tokenid[k],
                pos_ctx_tokenmask = caseinfo.sim_ctx_tokenmask[k],
                pos_img = caseinfo.sim_imgs[k]
                currcase = Pack(
                    caseid = caseinfo.caseid,  # int
                    topk_caseids = caseinfo.topk_caseids,
                    curr_ctx_feature = caseinfo.curr_ctx_feature,
                    curr_ctx_tokenid = caseinfo.curr_ctx_tokenid,
                    curr_ctx_tokenmask = caseinfo.curr_ctx_tokenmask,
                    curr_rsp_feature = caseinfo.curr_rsp_feature,
                    curr_rsp_tokenid = caseinfo.curr_rsp_tokenid,
                    curr_rsp_tokenmask = caseinfo.curr_rsp_tokenmask,
                    curr_img = caseinfo.curr_img,
                    
                    pos_ctx_feature = pos_ctx_feature,
                    pos_ctx_tokenid = pos_ctx_tokenid,
                    pos_ctx_tokenmask = pos_ctx_tokenmask,
                    pos_caseid = pos_caseid,
                    pos_img = caseinfo.curr_img,
                    
                    sim_ctx_feature = torch.stack(caseinfo.sim_ctx_feature),
                    sim_ctx_tokenid = torch.stack(caseinfo.sim_ctx_tokenid),
                    sim_ctx_tokenmask = torch.stack(caseinfo.sim_ctx_tokenmask),
                    sim_rsp_feature = torch.stack(caseinfo.sim_rsp_feature),
                    sim_rsp_tokenid = torch.stack(caseinfo.sim_rsp_tokenid),
                    sim_rsp_tokenmask = torch.stack(caseinfo.sim_rsp_tokenmask),
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
   
        cases = []
        topk_caseids = []
        curr_ctx_feature = []
        curr_ctx_tokenid = []
        curr_ctx_tokenmask = []
        curr_rsp_feature = []
        curr_rsp_tokenid = []
        curr_rsp_tokenmask = []
        curr_img = []
        sim_ctx_feature = []
        sim_ctx_tokenid = []
        sim_ctx_tokenmask = []
        sim_rsp_feature = []
        sim_rsp_tokenid = []
        sim_rsp_tokenmask = []
        sim_ids = []
        sim_imgs = []

        for idx in selected_index:
            cases.append(self.data_pairs[idx])
            topk_caseids.append(self.data_pairs[idx].topk_caseids)
            curr_ctx_feature.append(self.data_pairs[idx].curr_ctx_feature)
            curr_ctx_tokenid.append(self.data_pairs[idx].curr_ctx_tokenid)
            curr_ctx_tokenmask.append(self.data_pairs[idx].curr_ctx_tokenmask)
            curr_rsp_feature.append(self.data_pairs[idx].curr_rsp_feature)
            curr_rsp_tokenid.append(self.data_pairs[idx].curr_rsp_tokenid)
            curr_rsp_tokenmask.append(self.data_pairs[idx].curr_rsp_tokenmask)
            curr_img.append(self.data_pairs[idx].curr_img)
            sim_ctx_feature.append(self.data_pairs[idx].sim_ctx_feature)
            sim_ctx_tokenid.append(self.data_pairs[idx].sim_ctx_tokenid)
            sim_ctx_tokenmask.append(self.data_pairs[idx].sim_ctx_tokenmask)
            sim_rsp_feature.append(self.data_pairs[idx].sim_rsp_feature)
            sim_rsp_tokenid.append(self.data_pairs[idx].sim_rsp_tokenid)
            sim_rsp_tokenmask.append(self.data_pairs[idx].sim_rsp_tokenmask)
            sim_ids.append(self.data_pairs[idx].sim_ids)
            sim_imgs.append(self.data_pairs[idx].sim_imgs)

        max_dec_len = self.config.max_dec_len#50
        max_curr_ctx_len = self.max_curr_ctx_len
        max_sim_ctx_len = self.max_curr_ctx_len  
        batch_size = len(selected_index)  # number of pairs in this batch
        dim_img_in = cases[0].curr_img.size  
        dim_ctx_in = cases[0].curr_ctx_feature.size(1)
        
        
        caseids = [i.caseid for i in cases] 
        labels = caseids + caseids#BS*2 
        labels = np.array(labels, dtype=np.int32)


        curr_rsps_teach = []
        curr_rsps_target = []
        for batch_id, case in enumerate(cases):#list to array   
            curr_rep_teach_tokenid_sample=[]
            for i in case.curr_rsp_tokenid.cpu().tolist():
                if i!=0:
                    curr_rep_teach_tokenid_sample.append(i)
            curr_rsps_teach.append([t for _, t in enumerate(self.pad_to(max_dec_len, (curr_rep_teach_tokenid_sample)[:-2], do_pad=True))])#去掉了<eos>
            curr_rsps_target.append([t for _, t in enumerate(self.pad_to(max_dec_len, (case.curr_rsp_tokenid.cpu().tolist())[2:], do_pad=True))])#去掉了<s>



             

        return Pack( 
 
            
            caseid = torch.Tensor(caseids),
            topk_caseids_in = torch.from_numpy(np.array(topk_caseids)),
            curr_ctx_feature = torch.stack(curr_ctx_feature),
            curr_ctx_tokenid = torch.stack(curr_ctx_tokenid),
            curr_ctx_tokenmask = torch.stack(curr_ctx_tokenmask),
            curr_rsp_feature = torch.stack(curr_rsp_feature),
            curr_rsp_tokenid = torch.stack(curr_rsp_tokenid),
            curr_rsp_tokenmask = torch.stack(curr_rsp_tokenmask),
            curr_img = torch.from_numpy(np.array(curr_img)),
            
            curr_rsps_teach_tokenid = torch.Tensor(curr_rsps_teach),
            curr_rsps_target_tokenid = torch.Tensor(curr_rsps_target),

            sim_ctx_feature = torch.cat(sim_ctx_feature),
            sim_ctx_tokenid = torch.cat(sim_ctx_tokenid),
            sim_ctx_tokenmask = torch.cat(sim_ctx_tokenmask),
            sim_rsp_feature = torch.cat(sim_rsp_feature),
            sim_rsp_tokenid = torch.cat(sim_rsp_tokenid),
            sim_rsp_tokenmask = torch.cat(sim_rsp_tokenmask),
            
            sim_ids = torch.from_numpy(np.array(sim_ids)),
            sim_imgs = torch.from_numpy(np.array(sim_imgs)),
            
            batch_size =torch.from_numpy(np.array(batch_size))
            
            
        )

    
    def selected_batch(self, selected_ids):

        return self._prepare_batch(selected_index=selected_ids)


    # paired level
    def _shuffle_indexes_pairs(self):
        np.random.shuffle(self.indexes_pairs_shuffled)

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
            self.indexes_pairs = self.indexes_pairs_shuffled
        else:
            self.indexes_pairs = self.indexes_pairs_unshuffled

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
    def selected_batch_pairs(self, selected_ids):

        return self._prepare_batch_pairs(selected_index=selected_ids)               

    def _prepare_batch_pairs(self, selected_index):
#         ipdb.set_trace()
#         cases = [self.data_pairs[idx] for idx in selected_index] 
        cases = []
        topk_caseids = []
        curr_ctx_feature = []
        curr_ctx_tokenid = []
        curr_ctx_tokenmask = []
        curr_rsp_feature = []
        curr_rsp_tokenid = []
        curr_rsp_tokenmask = []
        curr_img = []
        sim_ctx_feature = []
        sim_ctx_tokenid = []
        sim_ctx_tokenmask = []
        sim_rsp_feature = []
        sim_rsp_tokenid = []
        sim_rsp_tokenmask = []
        sim_ids = []
        sim_imgs = []

        for idx in selected_index:
            cases.append(self.data_pairs[idx])
            topk_caseids.append(self.data_pairs[idx].topk_caseids)
            curr_ctx_feature.append(self.data_pairs[idx].curr_ctx_feature)
            curr_ctx_tokenid.append(self.data_pairs[idx].curr_ctx_tokenid)
            curr_ctx_tokenmask.append(self.data_pairs[idx].curr_ctx_tokenmask)
            curr_rsp_feature.append(self.data_pairs[idx].curr_rsp_feature)
            curr_rsp_tokenid.append(self.data_pairs[idx].curr_rsp_tokenid)
            curr_rsp_tokenmask.append(self.data_pairs[idx].curr_rsp_tokenmask)
            curr_img.append(self.data_pairs[idx].curr_img)
            sim_ctx_feature.append(self.data_pairs[idx].sim_ctx_feature)
            sim_ctx_tokenid.append(self.data_pairs[idx].sim_ctx_tokenid)
            sim_ctx_tokenmask.append(self.data_pairs[idx].sim_ctx_tokenmask)
            sim_rsp_feature.append(self.data_pairs[idx].sim_rsp_feature)
            sim_rsp_tokenid.append(self.data_pairs[idx].sim_rsp_tokenid)
            sim_rsp_tokenmask.append(self.data_pairs[idx].sim_rsp_tokenmask)
            sim_ids.append(self.data_pairs[idx].sim_ids)
            sim_imgs.append(self.data_pairs[idx].sim_imgs)
                    
        max_dec_len = self.config.max_dec_len#50
        max_curr_ctx_len = self.max_curr_ctx_len
        max_sim_ctx_len = self.max_curr_ctx_len  
        batch_size = len(selected_index)  # number of pairs in this batch
        dim_img_in = cases[0].curr_img.size  
        dim_ctx_in = cases[0].curr_ctx_feature.size(1)
        
        text_pair_in = np.zeros((batch_size * 2, max_curr_ctx_len, dim_ctx_in), dtype=np.float32)
        img_pair_in = np.zeros((batch_size * 2, dim_img_in), dtype=np.float32)
        
        caseids = [i.caseid for i in cases] 
        labels = caseids + caseids#BS*2 
        labels = np.array(labels, dtype=np.int32)
        
#         vec_curr_rsps_teach = np.zeros((batch_size, max_dec_len), dtype=np.int32)
#         vec_curr_rsps_target = np.zeros((batch_size, max_dec_len), dtype=np.int32)

        curr_rsps_teach = []
        curr_rsps_target = []
        for batch_id, case in enumerate(cases):#list to array   
            curr_rep_teach_tokenid_sample=[]
            for i in case.curr_rsp_tokenid.cpu().tolist():
                if i!=0:
                    curr_rep_teach_tokenid_sample.append(i)
            curr_rsps_teach.append([t for _, t in enumerate(self.pad_to(max_dec_len, (curr_rep_teach_tokenid_sample)[:-1], do_pad=True))])#去掉了<eos>
            curr_rsps_target.append([t for _, t in enumerate(self.pad_to(max_dec_len, (case.curr_rsp_tokenid.cpu().tolist())[1:], do_pad=True))])#去掉了<s>
        
            text_pair_in[batch_id, :] = case.curr_ctx_feature
            text_pair_in[batch_id + batch_size, :] = case.pos_ctx_feature 
            img_pair_in[batch_id, :] = case.curr_img  #array
            img_pair_in[batch_id + batch_size, :] = case.pos_img  #array 


             
#         ipdb.set_trace()
        return Pack( 
            text_pair_in = torch.from_numpy(text_pair_in), 
            img_pair_in = torch.from_numpy(img_pair_in),  
            labels = torch.from_numpy(labels),  
            topk_caseids_in = torch.from_numpy(np.array(topk_caseids)),
            caseid = torch.Tensor(caseids),
            curr_ctx_feature = torch.stack(curr_ctx_feature),
            curr_ctx_tokenid = torch.stack(curr_ctx_tokenid),
            curr_ctx_tokenmask = torch.stack(curr_ctx_tokenmask),
            curr_rsp_feature = torch.stack(curr_rsp_feature),
            curr_rsp_tokenid = torch.stack(curr_rsp_tokenid),
            curr_rsp_tokenmask = torch.stack(curr_rsp_tokenmask),
            curr_img = torch.from_numpy(np.array(curr_img)),
            
            curr_rsps_teach_tokenid = torch.Tensor(curr_rsps_teach),
            curr_rsps_target_tokenid = torch.Tensor(curr_rsps_target),

            sim_ctx_feature = torch.cat(sim_ctx_feature),
            sim_ctx_tokenid = torch.cat(sim_ctx_tokenid),
            sim_ctx_tokenmask = torch.cat(sim_ctx_tokenmask),
            sim_rsp_feature = torch.cat(sim_rsp_feature),
            sim_rsp_tokenid = torch.cat(sim_rsp_tokenid),
            sim_rsp_tokenmask = torch.cat(sim_rsp_tokenmask),
            
            sim_ids = torch.from_numpy(np.array(sim_ids)),
            sim_imgs = torch.from_numpy(np.array(sim_imgs)),
            
            batch_size =torch.from_numpy(np.array(batch_size))
            
            
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

