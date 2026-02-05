import os, sys
import csv
import json
import logging
import numpy as np
from collections import Counter
import ipdb
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
#         train_textvecs = np.loadtxt(CUR_PATH + '/../retrieve_text/textvec/train_textvecs.csv', dtype=float, delimiter='\t')#cur ctx vec n*768
#         train_texts = open(textfolder_path + '/unsup_text_train.txt', 'r').readlines()#文本vec  23765
        train_texts_path = CASE_DATA_PATH + '/bs_context_train.txt' if config.ctx_with_bs else CASE_DATA_PATH + '/context_train.txt'
        train_texts = open(train_texts_path, 'r').readlines()
        train_rsp_path = CASE_DATA_PATH + '/response_train.txt'
#         train_ctx_data = open(train_ctx_path, 'r').readlines()
        train_ctx_data = train_texts
        train_rsp_data = open(train_rsp_path, 'r').readlines()
         

        for mode in modes:
            mode_topk_path = gt_pair_path + '/' + mode + '_top10_didtid.json'  #3500个train_didtid2caseid相对应的十个label
            mode_topk_data = json.load(open(mode_topk_path))

            mode_didtid2caseid_path = vecfolder_path + '/' + mode + '_didtid2caseid.json'
            mode_caseid2didtid_path = vecfolder_path + '/' + mode + '_caseid2didtid.json'
            
            mode_imgname2vecid_path = vecfolder_path + '/' + mode + '_imgname2vecid.json'
            mode_imgvec_filepath = CUR_PATH + '/../retrieve_img/imgvec_nofc/' + mode + '_imgvecs.csv'
#             mode_textvec_filepath = CUR_PATH + '/../retrieve_text/textvec/' + mode + '_textvecs.csv'
#             mode_text_filepath = textfolder_path + '/unsup_text_' + mode + '.txt'
            mode_text_filepath = CASE_DATA_PATH + '/bs_context_' + mode + '.txt' if config.ctx_with_bs else CASE_DATA_PATH + '/context_' + mode + '.txt'
    
            mode_img_path = img_folder_path + '/'+mode

            mode_didtid2caseid = json.load(open(mode_didtid2caseid_path))
            mode_caseid2didtid = json.load(open(mode_caseid2didtid_path))
            mode_imgname2vecid = json.load(open(mode_imgname2vecid_path))
            mode_imgvecs = np.loadtxt(mode_imgvec_filepath, dtype=float, delimiter='\t')
#             mode_textvecs = np.loadtxt(mode_textvec_filepath, dtype=float, delimiter='\t')#cur ctx vec n*768
            mode_texts = open(mode_text_filepath, 'r').readlines() 



#             mode_ctx_path = CASE_DATA_PATH + '/bs_context_' + mode + '.txt' if config.ctx_with_bs else CASE_DATA_PATH + '/context_' + mode + '.txt'
#             mode_ctx_data = open(mode_ctx_path, 'r').readlines()
            mode_ctx_data = mode_texts
            mode_rsp_path = CASE_DATA_PATH + '/response_' + mode + '.txt'
            mode_rsp_data = open(mode_rsp_path, 'r').readlines()

            # get similar caseid
            if mode == 'train' and not config.train_retrieved_sim:
                mode_topk_caseid = json.load(open(DATA_PATH + '/groundtruth/' + mode + '_top10_caseid.json'))#与retrieval里面dataset的groundtruth一致
            else:
                if config.cluster:
                    mode_topk_caseid = json.load(open(DATA_PATH + '/clustering_results/' \
                      + mode + '_top' + str(config.cluster_k) \
                      + '_' + config.cluster_select + '.json'))
                else: #config.train_retrieved_sim==True
                    if not config.retrieve_results_path:
                        curr_retrieve_results_path = 'retrieve_results'
                    else:
                        curr_retrieve_results_path = config.retrieve_results_path   #'eval_candidate/seed_11'里面test和val可以通过retrieval获取，需要修改代码生成train部分
                    print('use correct corpus: current retrieve results folder: ' + curr_retrieve_results_path)
                    
                    mode_topk_caseid = json.load(open(DATA_PATH + '/' + curr_retrieve_results_path + '/' + mode + '_topk_best.json'))                    
            
            mode_data = self._process_vecs(dialogues, mode_topk_data, config.K,
                                           mode_didtid2caseid, mode_imgname2vecid, mode_caseid2didtid,
                                           train_didtid2caseid, train_imgname2vecid, train_caseid2didtid, 
                                           train_imgvecs, train_texts,
                                           mode_imgvecs, mode_texts,
                                           mode, config.num_sim, 
                                           mode_topk_caseid,
                                           mode_rsp_data,
                                           mode_ctx_data,
                                           train_ctx_data,
                                           train_rsp_data
                                          )
            results.append(mode_data)

        return results

    def _process_vecs(self, dialogues, mode_topk_data, K,
                                           mode_didtid2caseid, mode_imgname2vecid, mode_caseid2didtid,
                                           train_didtid2caseid, train_imgname2vecid, train_caseid2didtid, 
                                           train_imgvecs, train_texts,
                                           mode_imgvecs, mode_texts,
                                           mode, num_sim, 
                                           mode_topk_caseid,
                                           mode_rsp_data,
                                           mode_ctx_data,
                                           train_ctx_data,
                                           train_rsp_data
                     ):

        results = []
        num_case = len(mode_rsp_data)
        
        for did in mode_didtid2caseid:
            for tid in mode_didtid2caseid[did]:#从1开始
                caseid = mode_didtid2caseid[did][tid]#给定case
#                 pairs = mode_topk_data[did][tid]#相似10样本
#                 ipdb.set_trace()
#                 load current text and img vector


                turn_info = dialogues[did]['dialogue'][int(tid)-1]#取当前轮对话内容
                imgs = turn_info['user']['imgs'] + turn_info['agent']['imgs']
                imgvec = np.zeros(self.config.dim_img_in, dtype=float)#无图片则全0向量
                if len(imgs) != 0:
                    img_name = imgs[0]
                    imgvecid = mode_imgname2vecid[img_name]
                    imgvec = mode_imgvecs[imgvecid]                  
 
                curr_ctx = mode_texts[caseid].strip('\n')
                curr_ctx_id = self._sent2id(self.tokenize(curr_ctx))#tokenizer + 映射为id
                curr_rsp = mode_rsp_data[caseid].strip('\n')
                curr_rsp_id = self._sent2id(self.tokenize(curr_rsp))
                
                sim_caseids = mode_topk_caseid[caseid][:num_sim]
                topk_caseids = mode_topk_caseid[caseid][:K]
                sim_ctxs = []
                sim_ctxs_id = []
                sim_rsps = []
                sim_rsps_id = []
                sim_ids = []
                sim_imgs = []
                max_sim_ctx_len = 0
                max_sim_rsp_len = 0
                for sim_caseid in sim_caseids:
#                     ipdb.set_trace()
                    did2, tid2 = train_caseid2didtid[str(sim_caseid)]
                    sim_ctx = train_texts[sim_caseid].strip('\n')
                    sim_ctx_id = self._sent2id(self.tokenize(sim_ctx))
                    sim_rsp = train_rsp_data[sim_caseid].strip('\n')
                    sim_rsp_id = self._sent2id(self.tokenize(sim_rsp))
                    
                    sim_ctxs.append(sim_ctx)
                    sim_ctxs_id.append(sim_ctx_id)                    
                    sim_rsps.append(sim_rsp)
                    sim_rsps_id.append(sim_rsp_id)
                    
                    sim_ids.append(sim_caseid)
                    max_sim_ctx_len = max_sim_ctx_len if len(sim_ctx) < max_sim_ctx_len else len(sim_ctx)
                    max_sim_rsp_len = max_sim_rsp_len if len(sim_rsp) < max_sim_rsp_len else len(sim_rsp)                    
                    
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
                    
                    # list of 文本tokenizer后的id
                    caseid = caseid,
                    curr_ctx = curr_ctx,                    
                    curr_ctx_tokenid = curr_ctx_id,
                    curr_rsp=curr_rsp,
                    curr_rsp_tokenid=curr_rsp_id,
                    curr_img = imgvec,
                    
                    sim_ids = sim_ids,
                    sim_ctxs=sim_ctxs, 
                    sim_ctxs_tokenid = sim_ctxs_id,
                    sim_rsps=sim_rsps,
                    sim_rsps_tokenid=sim_rsps_id,                   
                    sim_imgs = sim_imgs,
                )
#                 ipdb.set_trace()
                results.append(currcase)
                
        return results


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

