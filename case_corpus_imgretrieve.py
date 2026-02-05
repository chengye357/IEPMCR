import os, sys
import csv
import json
import numpy as np
from collections import Counter
import json
import logging
import ipdb
from random import randrange
import torch as th
from pprg_end2end.utils import Pack, DATA_PATH
from pprg_end2end.case_util import *

logger = logging.getLogger()
def remove_img(text):
    if '_img' in text:
        text = text.replace('_img', '')
    return text

class CaseCorpus_imgretrieve(object):

    def __init__(self, config):
        self.config = config
        self.train_imgvecs = np.loadtxt(DATA_PATH + '/imgvec/train_imgvecs.csv', dtype=float, delimiter='\t')
        self.tokenize = lambda x: x.split()
        self.train_corpus, self.test_corpus = self._read_file(self.config)
        logger.info('Loading corpus finished.')
        logger.info('-'*20)


    def _read_file(self, config):
        results = []
        modes = ['train', 'test']

        dialogues = json.load(open(DATA_PATH + '/dialogues.json'))#对话

        train_imgname2vecid = json.load(open(DATA_PATH + '/vecfolder/train_imgname2vecid.json'))#
        train_imgvecs = np.loadtxt(DATA_PATH + '/imgvec/train_imgvecs.csv', dtype=float, delimiter='\t')
        train_didtid2caseid = json.load(open(DATA_PATH + '/vecfolder/train_didtid2caseid.json'))# case  3500
        train_caseid2didtid = json.load(open(DATA_PATH + '/vecfolder/train_caseid2didtid.json'))
        # train_rspvecs = np.loadtxt(config.saved_path + '/train_rspvec.csv', dtype=float, delimiter='\t')
        # train_weightvecs = np.loadtxt(config.saved_path + '/train_weightvec.csv', dtype=float, delimiter='\t')
        train_cluster_topk = th.load(DATA_PATH + '/cluster/sorted_ids.pt')
        train_im2cluster = th.load(DATA_PATH + '/cluster/im2cluster.pt')
#         ipdb.set_trace()
        saved_data_path = remove_img(config.saved_path)
                   
            
        for mode in modes:
            mode_imgname2vecid = json.load(open(DATA_PATH + '/vecfolder/' + mode + '_imgname2vecid.json'))
            mode_imgvecs = np.loadtxt(DATA_PATH + '/imgvec/' + mode + '_imgvecs.csv', dtype=float, delimiter='\t')

            mode_didtid2caseid = json.load(open(DATA_PATH + '/vecfolder/' + mode + '_didtid2caseid.json'))
            
            mode_rspvecs = np.loadtxt(saved_data_path + '/' + mode + '_rspvec.csv', dtype=float, delimiter='\t')
            mode_weightvecs = np.loadtxt(saved_data_path + '/' + mode + '_weightvec.csv', dtype=float, delimiter='\t')

            # get similar caseid
            if config.cluster:
                mode_topk_caseid = json.load(open(DATA_PATH + '/clustering_results/' \
                                                  + mode + '_top' + str(config.cluster_k) \
                                                  + '_' + config.cluster_select + '.json'))
            else:
                if not  config.train_retrieved_sim:
                    mode_topk_caseid = json.load(open(DATA_PATH + '/groundtruth/' + mode + '_top10_caseid.json'))
                else:
                    if not config.retrieve_results_path:
                        curr_retrieve_results_path = 'retrieve_results'
                    else:
                        curr_retrieve_results_path = config.retrieve_results_path
#                     print('use correct corpus: current retrieve results folder: ' + curr_retrieve_results_path + '/' + mode + '_topk_best.json')
#                     mode_topk_caseid = json.load(open(DATA_PATH + '/' + curr_retrieve_results_path + '/' + mode + '_topk_best.json'))  
                    print('use correct corpus: current retrieve results folder: ' + saved_data_path + '/id/' + mode + '_topk_best.json')
#                     mode_topk_caseid = json.load(open(saved_data_path + '/id/' + mode + '_topk_best.json')) #自己的结果
                    mode_topk_caseid = json.load(open(saved_data_path + '/id/' + '21_' + mode + '_topk.json'))

            
            
            mode_data = self._process_mode(mode, config.num_sim, mode_topk_caseid,dialogues,
                                           train_imgname2vecid, train_imgvecs,
                                           train_caseid2didtid,train_didtid2caseid,
                                           mode_imgname2vecid, mode_imgvecs, mode_didtid2caseid,
                                           mode_rspvecs, mode_weightvecs, train_cluster_topk,train_im2cluster)
            results.append(mode_data)

        return results

    def _process_mode(self, mode, num_sim, mode_topk_caseid, dialogues,
                      train_imgname2vecid, train_imgvecs, train_caseid2didtid,train_didtid2caseid,
                      mode_imgname2vecid, mode_imgvecs, mode_didtid2caseid, mode_rspvecs, mode_weightvecs, 
                      train_cluster_topk,train_im2cluster):


        results = []
        num_trainimgs = len(train_imgvecs)
#         ipdb.set_trace()
        count1 = 1
        count2 = 1
        for did in mode_didtid2caseid: #did str; tid str; caseid int; 
            for tid in mode_didtid2caseid[did]:
                tinfo = dialogues[did]['dialogue'][int(tid)]
                gt_rsp_imgs = tinfo['agent']['imgs']
                if len(gt_rsp_imgs) == 0:
                    continue
                caseid = mode_didtid2caseid[did][tid]
                rspvec = mode_rspvecs[caseid]
                weightvec = mode_weightvecs[caseid]

                weighted_sim_imgvec = np.full(128,1e-10, dtype=float)
                sim_caseids = mode_topk_caseid[caseid][:num_sim]
                
                
#                 sim_caseids = self.tri_turn_sim_case(num_sim, sim_caseids, train_caseid2didtid, train_didtid2caseid, dialogues)          
                

                for i, sim_caseid in enumerate(sim_caseids): # sim_caseids 全int
                    sim_did, sim_tid = train_caseid2didtid[str(sim_caseid)]#这里tid从1到len(dialogues[did]['dialogue'])-1；caseid str; did str; tid int;

                    sim_tinfo = dialogues[sim_did]['dialogue'][sim_tid]
                    sim_rsp_imgs = sim_tinfo['agent']['imgs']
                    if len(sim_rsp_imgs) == 0:
                        sim_imgvec = np.full(128, 1e-10,dtype=float)
                    else:
                        sim_rsp_img = sim_rsp_imgs[0]
                        sim_rsp_img_vecid = train_imgname2vecid[sim_rsp_img]
                        sim_imgvec = train_imgvecs[sim_rsp_img_vecid]
                        
                    weighted_sim_imgvec += weightvec[i] * sim_imgvec  #top5检索到的相似case, 其图片表征的加权结果
# triturn 替换图像                   
                if max(weighted_sim_imgvec) == min(weighted_sim_imgvec):
                    for i, sim_caseid in enumerate(sim_caseids): # sim_caseids 全int
                        sim_did, sim_tid = train_caseid2didtid[str(sim_caseid)]#这里tid从1到len(dialogues[did]['dialogue'])-1；caseid str; did str; tid int;

                        sim_imgvec = self.tri_turn_sim_img(sim_did, sim_tid, dialogues, train_imgname2vecid, train_imgvecs)
                        if max(sim_imgvec)== min(sim_imgvec):
                            count1+=1
                        else:
                            count2 +=1
                        weighted_sim_imgvec += weightvec[i] * sim_imgvec  #top5检索到的相似case, 其图片表征的加权结果
# #prototype 替换图像
#                 if max(weighted_sim_imgvec) == min(weighted_sim_imgvec):
#                     for i, sim_caseid in enumerate(sim_caseids): # sim_caseids 全int
#                         sim_did, sim_tid = train_caseid2didtid[str(sim_caseid)]#这里tid从1到len(dialogues[did]['dialogue'])-1；caseid str; did str; tid int;
#                         sim_imgvec = self.prototype_sim_img(sim_did, sim_tid, dialogues, train_imgname2vecid, train_imgvecs, train_didtid2caseid, train_caseid2didtid, train_cluster_topk, train_im2cluster)
#                         if max(sim_imgvec)== min(sim_imgvec):
#                             count1+=1
#                         else:
#                             count2 +=1
#                         weighted_sim_imgvec += weightvec[i] * sim_imgvec  #top5检索到的相似case, 其图片表征的加权结果

                
#                 ipdb.set_trace()
                if mode == 'train':
                    # build case for every true img
                    for gt_rsp_img in gt_rsp_imgs:
                        gt_rsp_img_vecid = mode_imgname2vecid[gt_rsp_img]
                        gt_imgvec = mode_imgvecs[gt_rsp_img_vecid]
                        for _rd in range(10):
                            rd_imgid = randrange(num_trainimgs)
                            rd_imgvec = train_imgvecs[rd_imgid]
                            currcase = Pack(
                                caseid=caseid,  # current id
                                rspvec=rspvec,  #600 rsp
                                weighted_sim_imgvec=weighted_sim_imgvec,#sim weight
                                gt_rsp_img=gt_rsp_img,#gt_rsp_img name
                                gt_imgvec=gt_imgvec,#128 
                                rd_imgvec=rd_imgvec#128 随机图片
                            )
                            results.append(currcase)
                else:
                    
                    currcase = Pack(
                        gt_rsp_img=gt_rsp_imgs,
                        caseid=caseid,  # int
                        rspvec=rspvec,
                        weighted_sim_imgvec=weighted_sim_imgvec
                    )
                    results.append(currcase)
        print(f"更改后仍无图比例：改后无图{count1} 改后有图{count2} 改前无图{count1+count2}")
        print(f"更换比例：{count2/(count1+count2)}")
        return results
    
    def tri_turn_sim_case(self, num_sim, sim_caseids, train_caseid2didtid, train_didtid2caseid, dialogues):
        for i in range(num_sim):#判断当前sim_case是否有图片
            sim_caseid = sim_caseids[i]
            imgs = []
            did1, tid1 = train_caseid2didtid[str(sim_caseid)]  #tid从1到len(dialogues[did]['dialogue'])-1
            turn_info = dialogues[did1]['dialogue'][int(tid1)]#tid从0到len(dialogues[did]['dialogue'])-1
            imgs = turn_info['user']['imgs'] + turn_info['agent']['imgs']                
            if len(imgs) == 0:
                turn_length = len(dialogues[did1]['dialogue'])
                print("turn_length:", turn_length)                
                if turn_length != 2:                
                    if tid1 == turn_length - 1:
                        tid1 = tid1 - 1
                        turn_info = dialogues[did1]['dialogue'][int(tid1)]#取t-1轮对话内容
                        imgs = turn_info['user']['imgs'] + turn_info['agent']['imgs']  
                        if len(imgs) != 0:                        
                            sim_caseids[i] = train_didtid2caseid[did1][str(tid1)] 
                    elif tid1 == 1:
                        tid1 = tid1 + 1 
                        turn_info = dialogues[did1]['dialogue'][int(tid1)]#取t-1轮对话内容
                        imgs = turn_info['user']['imgs'] + turn_info['agent']['imgs']  
                        if len(imgs) != 0:
                            sim_caseids[i] = train_didtid2caseid[did1][str(tid1)]
                    else:
                        tid11 = tid1 - 1
                        turn_info1 = dialogues[did1]['dialogue'][int(tid11)]#取t-1轮对话内容
                        imgs = turn_info1['user']['imgs'] + turn_info1['agent']['imgs']  
                        if len(imgs) != 0:
                            sim_caseids[i] = train_didtid2caseid[did1][str(tid11)]

                        else:
                            tid12 = tid1 + 1
                            turn_info2 = dialogues[did1]['dialogue'][int(tid12)]#取t+1轮对话内容
                            imgs = turn_info2['user']['imgs'] + turn_info2['agent']['imgs'] 
    #                                 ipdb.set_trace()
                            if len(imgs) != 0:
                                sim_caseids[i] = train_didtid2caseid[did1][str(tid12)]            
        return sim_caseids
    
    def is_img(self, sim_did, sim_tid, dialogues, train_imgname2vecid, train_imgvecs):
        tinfo = dialogues[sim_did]['dialogue'][sim_tid]#这里tid从0到len(dialogues[did]['dialogue'])-1;
        sim_rsp_imgs = tinfo['agent']['imgs']
        if len(sim_rsp_imgs) == 0:
            sim_imgvec = np.full(128, 1e-10,dtype=float)
            return sim_imgvec
        else:
            sim_rsp_img = sim_rsp_imgs[0]
            sim_rsp_img_vecid = train_imgname2vecid[sim_rsp_img]
            sim_imgvec = train_imgvecs[sim_rsp_img_vecid]
            return sim_imgvec
        
        
        
    def tri_turn_sim_img(self, sim_did, sim_tid, dialogues, train_imgname2vecid, train_imgvecs):        
        sim_tinfo = dialogues[sim_did]['dialogue'][sim_tid]#这里tid从0到len(dialogues[did]['dialogue'])-1;
        sim_rsp_imgs = sim_tinfo['agent']['imgs']
        if len(sim_rsp_imgs) == 0:
            sim_imgvec = np.full(128, 1e-10,dtype=float)
            turn_length = len(dialogues[sim_did]['dialogue'])              
            if turn_length != 2:                
                if sim_tid == turn_length - 1:
                    sim_tid = sim_tid - 1
                    sim_imgvec = self.is_img(sim_did, sim_tid, dialogues, train_imgname2vecid, train_imgvecs)
                elif sim_tid == 1:
                    sim_tid = sim_tid + 1 
                    sim_imgvec = self.is_img(sim_did, sim_tid, dialogues, train_imgname2vecid, train_imgvecs)
                else:
                    sim_tid1 = sim_tid - 1
                    sim_imgvec = self.is_img(sim_did, sim_tid1, dialogues, train_imgname2vecid, train_imgvecs)
                    if max(sim_imgvec) == min(sim_imgvec):
                        sim_tid2 = sim_tid + 1
                        sim_imgvec = self.is_img(sim_did, sim_tid2, dialogues, train_imgname2vecid, train_imgvecs)             
        else:
            sim_rsp_img = sim_rsp_imgs[0]
            sim_rsp_img_vecid = train_imgname2vecid[sim_rsp_img]
            sim_imgvec = train_imgvecs[sim_rsp_img_vecid]         
        return sim_imgvec
    
    def prototype_sim_img(self, sim_did, sim_tid, dialogues, train_imgname2vecid, train_imgvecs, train_didtid2caseid, train_caseid2didtid, train_cluster_topk, train_im2cluster):        
        sim_tinfo = dialogues[sim_did]['dialogue'][sim_tid]#这里tid从0到len(dialogues[did]['dialogue'])-1;
        sim_rsp_imgs = sim_tinfo['agent']['imgs']
        curr_case_id = train_didtid2caseid[sim_did][str(sim_tid)]
        curr_cluster_id = train_im2cluster[0][curr_case_id].item()
        k = 5
#         print(f"距聚类表征中心最近{k}个作为图片替换")
        curr_topk_sim_id = train_cluster_topk[0][curr_cluster_id][:k] #距聚类表征中心最近5个作为图片替换
#         ipdb.set_trace()
        sim_imgvec = np.full(128, 1e-10,dtype=float)
        
        for i in curr_topk_sim_id:
                         
            #拿出聚类id 
            
            sim_did1, sim_tid1 = train_caseid2didtid[str(int(i.item()))]
            sim_imgvec = self.is_img(sim_did1, sim_tid1, dialogues, train_imgname2vecid, train_imgvecs)
            
            if max(sim_imgvec) != min(sim_imgvec):
                break
        return sim_imgvec
      
    def get_corpus(self):
        return self.train_corpus, self.test_corpus


# if __name__ == "__main__":
#     config_path = "../../configs/pred_simctx_enc_fix_bs.conf"
#     config = Pack(json.load(open(config_path)))
#     config["forward_only"] = True
#     config["saved_path"] = '../../outputs/pred_simctx_enc_fix_bs'
#     corpus = CaseCorpus_imgretrieve(config)