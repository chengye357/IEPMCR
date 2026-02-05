import os, sys
import csv
import json
import logging
import numpy as np
from collections import Counter
import ipdb
import torch
from PIL import Image
# from transformers import AutoTokenizer
from pprg_end2end.utils import Pack, DATA_PATH
from pprg_end2end.case_util import *

CUR_PATH = os.path.join(os.path.dirname(__file__))
logger = logging.getLogger()

class CaseCorpus(object):

    def __init__(self, config):
        self.config = config
        self.tokenize = lambda x: x.split()
        vocab_file = DATA_PATH + '/vocab.json'
        if os.path.exists(vocab_file):
            with open(vocab_file, "r") as f:
                self.vocab_dict = json.load(f)
                self.vocab = list(self.vocab_dict.keys())
            self.unk_id = self.vocab_dict[UNK]
        else:
            self._extract_vocab()
            with open(vocab_file, "w") as f:
                json.dump(self.vocab_dict, f, ensure_ascii=False)
        logger.info('Loading vocab finished.')
        
        self.train_corpus, self.val_corpus, self.test_corpus = self._read_file(self.config)
        logger.info('Loading corpus finished.')
        logger.info('-'*20)
        
    def _extract_vocab(self):
        CASE_DATA_PATH = DATA_PATH + '/case_data'
        train_ctx_path = CASE_DATA_PATH + '/bs_context_train.txt' if self.config.ctx_with_bs else CASE_DATA_PATH + '/context_train.txt'
        train_ctx_data = open(train_ctx_path, 'r').readlines()

        all_words = []
        for ctx in train_ctx_data:
            tokens = self.tokenize(ctx.strip('\n'))
            all_words.extend(tokens)
        vocab_count = Counter(all_words).most_common()
        raw_vocab_size = len(vocab_count)
        use_vocab = SPECIAL_TOKENS + [t for t, cnt in vocab_count if t not in SPECIAL_TOKENS]
        self.vocab = use_vocab
        self.vocab_dict = {t: idx for idx, t in enumerate(self.vocab)}
        self.unk_id = self.vocab_dict[UNK]
        logger.info("Raw vocab size {} and final vocab size {}".format(raw_vocab_size, len(self.vocab)))
        
    def _read_file(self, config):
        CASE_DATA_PATH = DATA_PATH + '/case_data'
#         CTX_RSP_DATA_PATH = DATA_PATH + '/ctx_rsp_feature/'  
        CTX_RSP_DATA_PATH = DATA_PATH + '/ctx_rsp_feature_chen7200/'
        
        results = []
        modes = ['train', 'val', 'test']

        dialogue_path = DATA_PATH + '/dialogues_with_ctx.json'#对话
        vecfolder_path = DATA_PATH + '/vecfolder'##？？
        gt_pair_path = DATA_PATH + '/groundtruth'#
        textfolder_path = DATA_PATH + '/textfolder_withbs/all'#文本ctx文件夹
        img_folder_path = DATA_PATH + '/imgfolder' #图片png文件夹  
        
        dialogues = json.load(open(dialogue_path))
        train_didtid2caseid = json.load(open(vecfolder_path + '/train_didtid2caseid.json'))# case  3500
        train_caseid2didtid = json.load(open(vecfolder_path + '/train_caseid2didtid.json'))
        train_imgname2vecid = json.load(open(vecfolder_path + '/train_imgname2vecid.json'))#图像vec的name       3248
        train_imgvecs = np.loadtxt(CUR_PATH + '/../retrieve_img/imgvec_nofc/train_imgvecs.csv', dtype=float, delimiter='\t')#图像vec  3248*128
        
        train_ctx_data = torch.load(CTX_RSP_DATA_PATH + 'train' + '/ctx_vec.pt')
        train_ctx_tokenid_data = torch.load(CTX_RSP_DATA_PATH + 'train' + '/ctx_tokenid_vec.pt')
        train_ctx_tokenmask_data = torch.load(CTX_RSP_DATA_PATH + 'train' + '/ctx_tokenmask_vec.pt')
        train_rsp_data = torch.load(CTX_RSP_DATA_PATH + 'train' + '/rsp_vec.pt')
        train_rsp_tokenid_data = torch.load(CTX_RSP_DATA_PATH + 'train' + '/rsp_tokenid_vec.pt')
        train_rsp_tokenmask_data = torch.load(CTX_RSP_DATA_PATH + 'train' + '/rsp_tokenmask_vec.pt')

        saved_data_path = config.saved_path
        
        for mode in modes:
            mode_topk_path = gt_pair_path + '/' + mode + '_top10_didtid.json'  #3500个train_didtid2caseid相对应的十个label
            mode_topk_data = json.load(open(mode_topk_path))

            mode_didtid2caseid_path = vecfolder_path + '/' + mode + '_didtid2caseid.json'
            mode_caseid2didtid_path = vecfolder_path + '/' + mode + '_caseid2didtid.json'            
            mode_imgname2vecid_path = vecfolder_path + '/' + mode + '_imgname2vecid.json'
            mode_imgvec_filepath = CUR_PATH + '/../retrieve_img/imgvec_nofc/' + mode + '_imgvecs.csv'
            mode_didtid2caseid = json.load(open(mode_didtid2caseid_path))
            mode_caseid2didtid = json.load(open(mode_caseid2didtid_path))
            mode_imgname2vecid = json.load(open(mode_imgname2vecid_path))
            mode_imgvecs = np.loadtxt(mode_imgvec_filepath, dtype=float, delimiter='\t')
            
            mode_ctx_data = torch.load(CTX_RSP_DATA_PATH + mode + '/ctx_vec.pt')
            mode_ctx_tokenid_data = torch.load(CTX_RSP_DATA_PATH + mode + '/ctx_tokenid_vec.pt')
            mode_ctx_tokenmask_data = torch.load(CTX_RSP_DATA_PATH + mode + '/ctx_tokenmask_vec.pt')
            mode_rsp_data = torch.load(CTX_RSP_DATA_PATH + mode + '/rsp_vec.pt')
            mode_rsp_tokenid_data = torch.load(CTX_RSP_DATA_PATH + mode + '/rsp_tokenid_vec.pt')
            mode_rsp_tokenmask_data = torch.load(CTX_RSP_DATA_PATH + mode + '/rsp_tokenmask_vec.pt')

            # get similar caseid
            if mode == 'train' and not config.train_retrieved_sim:
                mode_topk_caseid = json.load(open(DATA_PATH + '/groundtruth/' + mode + '_top10_caseid.json'))#与retrieval里面dataset的groundtruth一致
                print('current retrieve results folder: groundtruth' )

            else:
                if config.cluster:
                    mode_topk_caseid = json.load(open(DATA_PATH + '/clustering_results/' \
                      + mode + '_top' + str(config.cluster_k) \
                      + '_' + config.cluster_select + '.json'))
                else: #config.train_retrieved_sim==True
                    if not config.retrieve_results_path:
                        curr_retrieve_results_path = 'retrieve_results'
                    else:
                        curr_retrieve_results_path = config.retrieve_results_path 
                    
#                     print('use correct corpus: current retrieve results folder: ' + curr_retrieve_results_path + '/' + mode + '_topk_best.json')
#                     mode_topk_caseid = json.load(open(DATA_PATH + '/' + curr_retrieve_results_path + '/' + mode + '_topk_best.json'))  
                    print('use correct corpus: current retrieve results folder: ' + saved_data_path + '/id/' + mode + '_topk_best.json')
#                     mode_topk_caseid = json.load(open(saved_data_path + '/id/' + mode + '_topk_best.json')) #自己的结果
                    mode_topk_caseid = json.load(open(saved_data_path + '/id/' + '21_' + mode + '_topk.json'))

            mode_data = self._process_vecs(dialogues, mode_topk_data, config.K,
                                           mode_didtid2caseid, mode_imgname2vecid, mode_caseid2didtid,
                                           train_didtid2caseid, train_imgname2vecid, train_caseid2didtid, 
                                           train_imgvecs,
                                           mode_imgvecs, 
                                           mode, 
                                           config.num_sim, 
                                           mode_topk_caseid,
                                           mode_ctx_data,
                                           mode_ctx_tokenid_data,
                                           mode_ctx_tokenmask_data,
                                           mode_rsp_data,
                                           mode_rsp_tokenid_data,
                                           mode_rsp_tokenmask_data,
                                           train_ctx_data,
                                           train_ctx_tokenid_data,
                                           train_ctx_tokenmask_data,
                                           train_rsp_data,
                                           train_rsp_tokenid_data,
                                           train_rsp_tokenmask_data
                                          )
            results.append(mode_data)

        return results

    def _process_vecs(self, dialogues, mode_topk_data, K,
                                           mode_didtid2caseid, mode_imgname2vecid, mode_caseid2didtid,
                                           train_didtid2caseid, train_imgname2vecid, train_caseid2didtid, 
                                           train_imgvecs,
                                           mode_imgvecs, 
                                           mode, 
                                           num_sim, 
                                           mode_topk_caseid,
                                           mode_ctx_data,
                                           mode_ctx_tokenid_data,
                                           mode_ctx_tokenmask_data,
                                           mode_rsp_data,
                                           mode_rsp_tokenid_data,
                                           mode_rsp_tokenmask_data,
                                           train_ctx_data,
                                           train_ctx_tokenid_data,
                                           train_ctx_tokenmask_data,
                                           train_rsp_data,
                                           train_rsp_tokenid_data,
                                           train_rsp_tokenmask_data
                     ):

        results = []
        num_case = len(mode_rsp_data)
        
        for did in mode_didtid2caseid:
            for tid in mode_didtid2caseid[did]:#从1开始
                caseid = mode_didtid2caseid[did][tid]#给定case

                turn_info = dialogues[did]['dialogue'][int(tid)-1]#取当前轮对话内容
                imgs = turn_info['user']['imgs'] + turn_info['agent']['imgs']
                curr_img = np.zeros(self.config.dim_img_in, dtype=float)#无图片则全0向量
                if len(imgs) != 0:
                    img_name = imgs[0]
                    imgvecid = mode_imgname2vecid[img_name]
                    curr_img = mode_imgvecs[imgvecid]
                  
 

                curr_ctx_feature = mode_ctx_data[caseid]
                curr_ctx_tokenid = mode_ctx_tokenid_data[caseid]
                curr_ctx_tokenmask = mode_ctx_tokenmask_data[caseid]
                curr_rsp_feature = mode_rsp_data[caseid]
                curr_rsp_tokenid = mode_rsp_tokenid_data[caseid]
                curr_rsp_tokenmask = mode_rsp_tokenmask_data[caseid]
                
                sim_caseids = mode_topk_caseid[caseid][:num_sim] #5
                topk_caseids = mode_topk_caseid[caseid][:K] #10

#                 sim_caseids, topk_caseids = self.tri_turn_sim_case(num_sim, sim_caseids, topk_caseids, train_caseid2didtid, train_didtid2caseid, dialogues)              
                            


                        
                
                sim_ctx_feature = []
                sim_ctx_tokenid = []
                sim_ctx_tokenmask = []
                sim_rsp_feature = []
                sim_rsp_tokenid = []
                sim_rsp_tokenmask = []
                sim_ids = []
                sim_imgs = []

                for sim_caseid in sim_caseids:

                    did2, tid2 = train_caseid2didtid[str(sim_caseid)]
               
                    sim_ctx_feature_sample = train_ctx_data[sim_caseid]
                    sim_ctx_tokenid_sample = train_ctx_tokenid_data[sim_caseid]
                    sim_ctx_tokenmask_sample = train_ctx_tokenmask_data[sim_caseid]
                    
                    sim_rsp_feature_sample = train_rsp_data[sim_caseid]
                    sim_rsp_tokenid_sample = train_rsp_tokenid_data[sim_caseid]
                    sim_rsp_tokenmask_sample = train_rsp_tokenmask_data[sim_caseid]
                    
                    sim_ctx_feature.append(sim_ctx_feature_sample)
                    sim_ctx_tokenid.append(sim_ctx_tokenid_sample)
                    sim_ctx_tokenmask.append(sim_ctx_tokenmask_sample)
                    sim_rsp_feature.append(sim_rsp_feature_sample)
                    sim_rsp_tokenid.append(sim_rsp_tokenid_sample)
                    sim_rsp_tokenmask.append(sim_rsp_tokenmask_sample)

                    sim_ids.append(sim_caseid)
                   
                    
                    imgvec2 = np.zeros(self.config.dim_img_in, dtype=float)
                    turn_info2 = dialogues[did2]['dialogue'][int(tid2) - 1]
                    imgs2 = turn_info2['user']['imgs'] + turn_info2['agent']['imgs']
                    if len(imgs2) != 0:
                        img_name2 = imgs2[0]
                        imgvecid2 = train_imgname2vecid[img_name2]
                        imgvec2 = train_imgvecs[imgvecid2]
                    sim_imgs.append(imgvec2) 
#                     print(f'2 {did2}, {tid2}, {imgvec2}')
                    

                    
                    
                currcase = Pack(
                    caseid = caseid,  # int
                    topk_caseids = topk_caseids,  # list of int, K
                    
                    curr_ctx_feature = curr_ctx_feature,
                    curr_ctx_tokenid = curr_ctx_tokenid,
                    curr_ctx_tokenmask = curr_ctx_tokenmask,
                    curr_rsp_feature = curr_rsp_feature,
                    curr_rsp_tokenid = curr_rsp_tokenid,
                    curr_rsp_tokenmask = curr_rsp_tokenmask,
                    curr_img = curr_img,
                    sim_ctx_feature = sim_ctx_feature,
                    sim_ctx_tokenid = sim_ctx_tokenid,
                    sim_ctx_tokenmask = sim_ctx_tokenmask,
                    sim_rsp_feature = sim_rsp_feature,
                    sim_rsp_tokenid = sim_rsp_tokenid,
                    sim_rsp_tokenmask = sim_rsp_tokenmask,
                    sim_ids = sim_ids,
                    sim_imgs = sim_imgs

                )
                results.append(currcase)
                
        return results
    
#     def tri_turn_sim_case(self, num_sim, sim_caseids, topk_caseids, train_caseid2didtid, train_didtid2caseid, dialogues):

#         for i in range(num_sim):#判断当前sim_case是否有图片
#             sim_caseid = sim_caseids[i]
#             imgs = []
#             did1, tid1 = train_caseid2didtid[str(sim_caseid)]
#             turn_info = dialogues[did1]['dialogue'][int(tid1)-1]#取当前轮对话内容
#             imgs = turn_info['user']['imgs'] + turn_info['agent']['imgs']                
#             if len(imgs) == 0:
#                 turn_length = len(dialogues[did1]['dialogue'])
#                 if tid1 == turn_length-1:
#                     tid1 = tid1 - 1
#                     turn_info = dialogues[did1]['dialogue'][int(tid1)-1]#取t-1轮对话内容
#                     imgs = turn_info['user']['imgs'] + turn_info['agent']['imgs']  
#                     if len(imgs) != 0:
#                         sim_caseids[i] = train_didtid2caseid[did1][str(tid1)]
#                         topk_caseids[i] = train_didtid2caseid[did1][str(tid1)]
#                 elif tid1 == 1:
#                     tid1 = tid1 + 1 
#                     turn_info = dialogues[did1]['dialogue'][int(tid1)-1]#取t-1轮对话内容
#                     imgs = turn_info['user']['imgs'] + turn_info['agent']['imgs']  
#                     if len(imgs) != 0:
#                         sim_caseids[i] = train_didtid2caseid[did1][str(tid1)]
#                         topk_caseids[i] = train_didtid2caseid[did1][str(tid1)]
#                 else:
#                     tid11 = tid1 - 1
#                     turn_info1 = dialogues[did1]['dialogue'][int(tid11)-1]#取t-1轮对话内容
#                     imgs = turn_info1['user']['imgs'] + turn_info1['agent']['imgs']  
#                     if len(imgs) != 0:
#                         sim_caseids[i] = train_didtid2caseid[did1][str(tid11)]
#                         topk_caseids[i] = train_didtid2caseid[did1][str(tid11)]
#                     else:
#                         tid12 = tid1 + 1
#                         turn_info2 = dialogues[did1]['dialogue'][int(tid12)-1]#取t+1轮对话内容
#                         imgs = turn_info2['user']['imgs'] + turn_info2['agent']['imgs'] 
# #                                 ipdb.set_trace()
#                         if len(imgs) != 0:
#                             sim_caseids[i] = train_didtid2caseid[did1][str(tid12)]
#                             topk_caseids[i] = train_didtid2caseid[did1][str(tid12)]
            
#         return sim_caseids, topk_caseids
                
                
    def get_corpus(self):
        return self.train_corpus, self.val_corpus, self.test_corpus

    def _sent2id(self, sent):
#         ipdb.set_trace()
        return [self.vocab_dict.get(t, self.unk_id) for t in sent]

    def id2sent(self, id_list):
        return [self.vocab[i] for i in id_list]

    def pad_to(self, max_len, tokens, do_pad):
        if len(tokens) >= max_len:
            return tokens[: max_len-1] + [tokens[-1]]
        elif do_pad:
            return tokens + [0] * (max_len - len(tokens))
        else:
            return tokens

