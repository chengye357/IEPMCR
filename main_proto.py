import os
import sys
import json
import numpy as np
import torch as th
from torch import nn
import ipdb
from datetime import datetime
from collections import defaultdict
import logging
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from pprg_end2end.utils import TBLogger, idx2word, Pack, DATA_PATH
import faiss

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score, calinski_harabasz_score
import shutil


PREVIEW_NUM = 0
logger = logging.getLogger()  

class LossManager(object):
    def __init__(self):
        self.losses = defaultdict(list)
        self.backward_losses = []

    def add_loss(self, loss):
        for key, val in loss.items():
            # print('key = %s\nval = %s' % (key, val))
            if val is not None and type(val) is not bool:
                self.losses[key].append(val.item())  # {key: [val1_scalar, val2_scalar...]} ?

    def pprint(self, name, window=None, prefix=None):
        str_losses = []
        for key, loss in self.losses.items():
            if loss is None:
                continue
            aver_loss = np.average(loss) if window is None else np.average(loss[-window:])
#             if 'nll' in key:  # TODO: ?
#                 str_losses.append('{} PPL {}'.format(key, np.exp(aver_loss)))
#             else:
            str_losses.append('{} {}'.format(key, aver_loss))

        if prefix:
            return '{}: {} {}'.format(prefix, name, ' '.join(str_losses))
        else:
            return '{} {}'.format(name, ' '.join(str_losses))

    def clear(self):
        self.losses = defaultdict(list)
        self.backward_losses = []

    def add_backward_loss(self, loss):
        self.backward_losses.append(loss.item())

    def avg_loss(self):
        return np.mean(self.backward_losses)

def lr_scheduler(epoch, optimizer, lr_decay_iter, decay_rate):
    if not (epoch % lr_decay_iter):
        for i in range(len(optimizer.param_groups)):
            optimizer.param_groups[i]['lr'] = optimizer.param_groups[i]['lr'] * decay_rate

def process_rerank(train_data, batch, topk_file, config):
    curr_id = batch.caseid.int().tolist() #tensor B
    sim_ids = []
    for i in curr_id:
        sim_id = topk_file[i]
        sim_ids.append(sim_id[:config.num_sim])
    sim_ids_reshape = [token for st in sim_ids for token in st]
    rerank_batch = train_data.selected_batch_pairs(sim_ids_reshape)
               
    batch.sim_ctx_feature = rerank_batch.curr_ctx_feature
    batch.sim_ctx_tokenid = rerank_batch.curr_ctx_tokenid
    batch.sim_ctx_tokenmask = rerank_batch.curr_ctx_tokenmask
    batch.sim_rsp_feature = rerank_batch.curr_rsp_feature
    batch.sim_rsp_tokenid = rerank_batch.curr_rsp_tokenid
    batch.sim_rsp_tokenmask = rerank_batch.curr_rsp_tokenmask
    batch.sim_ids = th.Tensor(sim_ids).long()
    batch.sim_imgs = rerank_batch.curr_img.view([-1, config.num_sim, config.dim_img_in])
    
    return batch

# def compute_distortion(results, x):
#     distortion_list = []
#     for centroids in results['centroids']:
#         i = 0
#         centroids = centroids.cpu().numpy()
#         distortion = 0
# #         ipdb.set_trace()
#         for j, centroid in enumerate(centroids):
            
#             indices = (results['im2cluster'][i] == j).cpu().numpy()
#             centroid_data = x[indices]
#             if len(centroid_data) > 0:  # 确保聚类中有数据点
#                 distortion += np.sum(np.linalg.norm(centroid_data - centroid, axis=1))
#         distortion_list.append(distortion)
#         i+=1
#     return distortion_list

def compute_distortion_sklearn(results, x):
    distortion_list = []
    for centroids in results['centroids']:
        i = 0
        distortion = 0
        for j, centroid in enumerate(centroids):
            
            indices = (results['im2cluster'][i] == j)
            centroid_data = x[indices]
            if len(centroid_data) > 0:  # 确保聚类中有数据点
                distortion += np.sum(np.linalg.norm(centroid_data - centroid, axis=1))
        distortion_list.append(distortion)
        i+=1
        
    print("计算distortion完成")
    return distortion_list

def compute_distortion(results, x):
    distortion_list = []
    for centroids in results['centroids']:
        i = 0
        centroids = centroids.cpu().numpy()
        distortion = 0
        for j, centroid in enumerate(centroids):
            indices = (results['im2cluster'][i] == j).cpu().numpy()
            centroid_data = x[indices]
            if len(centroid_data) > 0:  # 确保聚类中有数据点
                errors = centroid_data - centroid
                squared_errors = np.sum(errors ** 2, axis=1)
                distortion += np.sum(squared_errors)
        distortion_list.append(distortion)
        i += 1
    return distortion_list

def plot_tsne(M, J, im2cluster):
    plt.clf()
    # t-SNE可视化聚类结果
    print('Starting compute t-SNE Embedding...')
    print('分簇的数目: %d' % J)
    # 降维到2D用于绘图
    ts_2D = TSNE(n_components=2, perplexity=20, init='pca', random_state=0)
    res_2D = ts_2D.fit_transform(M)

    # 调用函数，绘制图像
#     plt.figure(1)
    plt.figure(figsize=(16,16))
#     plt.subplot(121)
    plt.scatter(res_2D[:, 0], res_2D[:, 1], c=im2cluster, s=1)
    plt.colorbar()


    plt.savefig(f'pprg_end2end/tsne/{J}_figure_tsne.png')

def plot_bar(J, im2cluster):
    plt.clf()
    # 统计每个聚类标签的数量
    label_counts = np.bincount(im2cluster)

    # 获取聚类标签的降序排列索引
    sorted_indices = np.argsort(label_counts)[::-1]

    # 根据排序索引对聚类标签和数量进行排序
    sorted_labels = np.array(sorted_indices)
    sorted_counts = label_counts[sorted_indices]

    # 生成聚类标签的横坐标
    x = np.arange(len(sorted_labels))

    # 绘制柱状图
    plt.bar(x, sorted_counts)

    # 设置横坐标刻度和标签
    plt.xticks(x, sorted_labels)

    # 在图例中显示聚类数量统计数据
    for i, count in enumerate(sorted_counts):
        plt.text(x[i], count, str(count), ha='center', va='bottom')

    # 设置图标题和坐标轴标签
    plt.title('Cluster Results')
    plt.xlabel('Cluster Label')
    plt.ylabel('Count')


    # 显示图形
    plt.savefig(f'pprg_end2end/tsne/{J}_figure_bar.png')
    
def plot_line(x_data, y_data):
    plt.clf()
    plt.plot(x_data, y_data, 'bo-')
    plt.xlabel('Number of clusters (k)')
    plt.ylabel('Distortion')
    plt.title('Elbow Method')
    plt.savefig('pprg_end2end/tsne/figure.png')
    
def generate_cluster(model, train_data, config):
    #生成原型聚类结果
    # models
    model.eval()

    logger.info('***** generate_cluster Begins at {} *****'.format(datetime.now().strftime("%Y-%m-%d %H-%M-%S")))
    config.num_cluster = config.num_cluster.split(',')
    config.num_cluster = list(map(int, config.num_cluster ))
    f_im2cluster = os.path.join(config.cluster_saved_path, 'im2cluster.pt')
    f_centroids = os.path.join(config.cluster_saved_path, 'centroids.pt')
    f_density = os.path.join(config.cluster_saved_path, 'density.pt')
    f_sorted_ids = os.path.join(config.cluster_saved_path, 'sorted_ids.pt')
    
    #cluter
    cluster_result = None
    # compute momentum features for center-cropped images
    features = compute_features(train_data, model, config)       #1、获取全部特征

    # placeholder for clustering result
#     cluster_result = {'im2cluster':[],'centroids':[],'density':[], 'sorted_ids':[]}
#     for num_cluster in config.num_cluster:
#         cluster_result['im2cluster'].append(th.zeros(features.shape[0],dtype=th.long).cuda())
#         cluster_result['centroids'].append(th.zeros(int(num_cluster),features.shape[1]).cuda())
#         cluster_result['density'].append(th.zeros(int(num_cluster)).cuda()) 
#         cluster_result['sorted_ids'].append(th.zeros(int(num_cluster)).cuda()) 

    features[th.norm(features,dim=1)>1.5] /= 2 #account for the few samples that are computed twice  
    features = features.numpy()

    cluster_result = run_kmeans(features,config)  #run kmeans clustering on master node
    ipdb.set_trace()
    th.save(cluster_result['im2cluster'], f_im2cluster)
    th.save(cluster_result['centroids'], f_centroids)
    th.save(cluster_result['density'], f_density)
    th.save(cluster_result['sorted_ids'], f_sorted_ids)



    print("kmeans聚类完成") 

        
def train(model, train_data, val_data, test_data, config, evaluator, evaluator_retrieval):
    # tensorboard
    tb_path = os.path.join(config.saved_path, "tensorboard/")
    tb_logger = TBLogger(tb_path)

    # training parameters
    batch_cnt, best_epoch_score = 0, 0
    best_score = -np.inf
    best_score_test = -np.inf
    train_loss = LossManager()
#     ipdb.set_trace()
    # models
    model.train()  # activate batch normalization and dropout
    optimizer = model.get_optimizer(config, verbose=False)
    
    EPOCH = 3 * config.lr_decay + 1
    ususe_rerank = config.ususe_rerank

    logger.info('***** Training Begins at {} *****'.format(datetime.now().strftime("%Y-%m-%d %H-%M-%S")))
    logger.info('***** Epoch 1/{} *****'.format(EPOCH))
    config.num_cluster = config.num_cluster.split(',')
    config.num_cluster  = list(map(int, config.num_cluster ))
    
    
    for epoch in range(1, EPOCH+1):
        
        

#         get updated sim_case_id         
        if ususe_rerank and epoch>1:
            last_train_topk = open(os.path.join(config.saved_path, 'id', '{}_train_topk.json'.format(epoch-1)), 'r')
            last_val_topk = open(os.path.join(config.saved_path, 'id', '{}_val_topk.json'.format(epoch-1)), 'r')
            last_test_topk = open(os.path.join(config.saved_path, 'id', '{}_test_topk.json'.format(epoch-1)), 'r')
            flast_train_topk = json.load(last_train_topk)
            flast_val_topk = json.load(last_val_topk)
            flast_test_topk = json.load(last_test_topk)
        else:
            flast_train_topk = None
            flast_val_topk = None
            flast_test_topk = None
            
            
            
            
        #cluster
        cluster_result = None
        # compute momentum features for center-cropped images
        features = compute_features(train_data, model, config)       #1、获取全部特征
        
        # placeholder for clustering result
        cluster_result = {'im2cluster':[],'centroids':[],'density':[]}
        for num_cluster in config.num_cluster:
            cluster_result['im2cluster'].append(th.zeros(features.shape[0],dtype=th.long).cuda())
            cluster_result['centroids'].append(th.zeros(int(num_cluster),features.shape[1]).cuda())
            cluster_result['density'].append(th.zeros(int(num_cluster)).cuda()) 

        features[th.norm(features,dim=1)>1.5] /= 2 #account for the few samples that are computed twice  
        features = features.numpy()
        
        cluster_result = run_kmeans(features,config)  #run kmeans clustering on master node
        print("kmeans聚类完成") 
        

                #lr decay
        model.train()  
        lr_scheduler(epoch, optimizer, config.lr_decay, config.lrd_rate)
        lr = optimizer.state_dict()['param_groups'][0]['lr']
        logger.info(f'lr\t:{lr}\t ')

            
        # EPOCH
        # for each epoch, reinitialize batches
        train_data.epoch_init_pairs(config, shuffle=True, verbose=epoch==0, drop_last=False)
        num_batch = train_data.num_batch_pairs

        while True:
            # BATCH
            batch = train_data.next_batch_pairs()#batch-pairs 
            batch_cnt += 1
#             ipdb.set_trace()
            if batch is None or batch.batch_size<config.batch_size:
                break
                
            if ususe_rerank:  
                if epoch>1:
                    batch = process_rerank(train_data, batch, flast_train_topk, config)#只替换train data里面的sim case


            optimizer.zero_grad()
            loss = model(batch, cluster_result) #这里应该有cluster_result
            train_loss.add_loss(loss)
            model.backward(loss, batch_cnt)
            nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)  # learnable parameters
            optimizer.step()
           
            

            # tensorboard save 
            data_dict = {}
            for key, val in loss.items():
                if val is not None and type(val) is not bool:
                    data_dict["train/%s"%key] = val.item()
            tb_logger.add_scalar_summary(data_dict, batch_cnt)

            # print training loss every print_frequency batch
            if batch_cnt % config.print_frequency == 0:
                logger.info(
                    train_loss.pprint(
                        'Train',
                        window=config.print_frequency,
                        prefix='{}/{}'.format(batch_cnt%num_batch,num_batch)))
                sys.stdout.flush()

        # Evaluate at the end of every epoch
        logger.info('Checkpoint step at {}'.format(datetime.now().strftime("%Y-%m-%d %H-%M-%S")))
        logger.info('==== Evaluating Model on Validation Set ====')

        if ususe_rerank: 
#             Generation topk_sim_id for next epoch
            train_vec_path = os.path.join(config.saved_path, 'vec', '{}_train_vec.csv'.format(epoch))
            ftrain_vec = open(train_vec_path, 'w')
            ftrain_topk = open(os.path.join(config.saved_path, 'id', '{}_train_topk.json'.format(epoch)), 'w')
            train_vecs = save_vec("train", flast_train_topk, flast_val_topk, flast_test_topk, model, config, train_data, ftrain_vec)
            p1_train, p3_train, p5_train, p10_train = save_vec_train("train", flast_train_topk, flast_val_topk, flast_test_topk, model, config, train_data, ftrain_vec, train_vecs=train_vecs, evaluator=evaluator_retrieval, topk_file=ftrain_topk)
            ftrain_topk.close()

            val_vec_path = os.path.join(config.saved_path, 'vec', '{}_val_vec.csv'.format(epoch))
            fval_vec = open(val_vec_path, 'w')
            fval_topk = open(os.path.join(config.saved_path, 'id', '{}_val_topk.json'.format(epoch)), 'w')
            p1_val, p3_val, p5_val, p10_val = save_vec("val", flast_train_topk, flast_val_topk, flast_test_topk, model, config, val_data, fval_vec, train_vecs=train_vecs, evaluator=evaluator_retrieval, train_data=train_data, topk_file=fval_topk)
            fval_topk.close()

            # just print test evaluation results
            test_vec_path = os.path.join(config.saved_path, 'vec', '{}_test_vec.csv'.format(epoch))
            ftest_vec = open(test_vec_path, 'w')
            ftest_topk = open(os.path.join(config.saved_path, 'id', '{}_test_topk.json'.format(epoch)), 'w')
            p1_test, p3_test, p5_test, p10_test = save_vec("test", flast_train_topk, flast_val_topk, flast_test_topk, model, config, test_data, ftest_vec, train_vecs=train_vecs, evaluator=evaluator_retrieval, train_data=train_data, topk_file=ftest_topk)
            ftest_topk.close()
        
            stats_train = {'train/p@1': p1_train, 'train/p@3': p3_train, 'train/p@5': p5_train, 'train/p@10': p10_train}
            tb_logger.add_scalar_summary(stats_train, batch_cnt)

            stats_val = {'val/p@1': p1_val, 'val/p@3': p3_val, 'val/p@5': p5_val, 'val/p@10': p10_val}
            tb_logger.add_scalar_summary(stats_val, batch_cnt)

            stats_test = {'test/p@1': p1_test, 'test/p@3': p3_test, 'test/p@5': p5_test, 'test/p@10': p10_test}
            tb_logger.add_scalar_summary(stats_test, batch_cnt)


        # Generation for reuse
        valid_loss, bleu = generate('val', epoch, model, val_data, config, evaluator,
                                    dest_f=open(os.path.join(config.saved_path, '{}_valid_file.txt'.format(epoch)), 'w'), train_data=train_data, last_topk_file=flast_val_topk)
        logger.info(train_loss.pprint('train'))
        stats = {'val/bleu': bleu, "val/loss": valid_loss}
        tb_logger.add_scalar_summary(stats, batch_cnt)

        test_loss, test_bleu = generate('test', epoch, model, test_data, config, evaluator,
                                        dest_f=open(os.path.join(config.saved_path, '{}_test_file.txt'.format(epoch)), 'w'),train_data=train_data, last_topk_file=flast_test_topk)
        logger.info(train_loss.pprint('train'))
        stats = {'test/bleu': test_bleu, "test/loss": test_loss}
        tb_logger.add_scalar_summary(stats, batch_cnt)
        if test_bleu > best_score_test:
            best_score_test = test_bleu
            best_epoch_score_test = epoch

        score = bleu
        cur_time = datetime.now().strftime("%Y-%m-%d %H-%M-%S")
        logger.info('*** Model Saved with val bleu = {:.3f}, test bleu = {:.3f} . ***'.format(bleu * 100, test_bleu * 100))
        model.save(config.saved_path, epoch)
        if score > best_score:
            best_epoch_score = epoch
            best_score = score
            model.save(config.saved_path, 'best')
            

        
        # exit val mode
        model.train()
        train_loss.clear()
        logger.info('Best validation score = %f at epoch %d' % (best_score, best_epoch_score))
        logger.info('Best test score = %f at epoch %d' % (best_score_test, best_epoch_score_test))
        logger.info('\n***** Epoch {}/{} *****'.format(epoch+1, EPOCH))

        sys.stdout.flush()

    logger.info('Training Ends.')
    return best_epoch_score


def generate(mode, epoch, model, data, config, evaluator, verbose=True, dest_f=None, train_data=None, last_topk_file=None):
    #生成新response
    
    model.eval()
    batch_cnt = 0

    data.epoch_init_pairs(config, shuffle=False, verbose=False, drop_last=False)  # all turns in a dialogue as a batch
    logger.debug('Generation: {} batches'.format(data.num_batch_pairs))
    losses = LossManager()
    
    all_preds = []
    all_labels = []

    while True:#update-topk-sim-case
        batch = data.next_batch_pairs()#dict_keys(['curr_ctxs', 'curr_rsps', 'curr_rsps_teach', 'curr_rsps_target', 'sim_ctxs', 'sim_rsps'])
        batch_cnt += 1
        if batch is None:
            break
        if epoch >1 :
            batch = process_rerank(train_data, batch, last_topk_file, config)
#         ipdb.set_trace()
        with th.no_grad():
            loss, outputs, labels = model.inference(batch)#output为pred的句词id,labels为真实词id
        losses.add_loss(loss)
        losses.add_backward_loss(model.model_sel_loss(loss, batch_cnt))  # a (weighted) sum of loss

        # move from GPU to CPU
        labels = labels.cpu()
        pred_labels = [t.cpu().data.numpy() for t in outputs]
        pred_labels = np.array(pred_labels, dtype=int)  # (batch_size, max_dec_len)
        true_labels = labels.data.numpy()  # (batch_size, output_seq_len)

        batch_size = pred_labels.shape[0]
        for b_id in range(batch_size):
            pred_str = idx2word(model.vocab, pred_labels, b_id) # return detokenized result
            true_str = idx2word(model.vocab, true_labels, b_id)
            all_preds.append([pred_str])
            all_labels.append([true_str])  # {filename: [true_responses, ...]}

            if verbose and batch_cnt <= PREVIEW_NUM:
                logger.debug('True: {}'.format(true_str, ))
                logger.debug('Pred: {}'.format(pred_str, ))
                logger.debug('-' * 40)

            if dest_f is not None:
                dest_f.write('True Rsp: {}\n'.format(true_str, ))
                dest_f.write('Pred Rsp: {}\n'.format(pred_str, ))
                dest_f.write('-' * 40+"\n")

    mode_loss = losses.avg_loss()  # mean of self.backward_losses
    logger.info(losses.pprint(mode))
    logger.info('--- Total loss = {}'.format(mode_loss))
    sys.stdout.flush()

    # data.name = 'Test', generated_dialogs = {filename: [pred_responses, ...]}, real_dialogs = {filename: [true_responses, ...]}
    task_report, bleu  = evaluator.evaluate(mode, all_preds, all_labels)

    logger.debug('Generation Done')
    logger.info(task_report)
    logger.debug('-' * 40)
    return mode_loss, bleu

def generate_img_recall(mode, model, data, config, evaluator, verbose=True, dest_f=None):
    model.eval()
    batch_cnt = 0

    data.epoch_init(config, shuffle=False, verbose=False, drop_last=False)  # all turns in a dialogue as a batch
    logger.debug('Generation: {} batches'.format(data.num_batch))

    train_vecs = np.loadtxt(DATA_PATH + '/imgvec/train_imgvecs.csv', dtype=float, delimiter='\t')
    test_vecs = np.loadtxt(DATA_PATH + '/imgvec/test_imgvecs.csv', dtype=float, delimiter='\t')
    # all_vecs = np.concatenate((train_vecs, test_vecs), axis=0)
    # train_vecid2imgname = json.load(open(DATA_PATH + '/vecfolder/train_vecid2imgname.json'))
    train_imgname2vecid = json.load(open(DATA_PATH + '/vecfolder/train_imgname2vecid.json'))
    test_imgname2vecid = json.load(open(DATA_PATH + '/vecfolder/test_imgname2vecid.json'))
    img2imgs = json.load(open(DATA_PATH + '/imgrspfolder/img2imgs.json'))
    test_caseid2didtid = json.load(open(DATA_PATH + '/vecfolder/test_caseid2didtid.json'))
    dialogues = json.load(open(DATA_PATH + '/dialogues.json'))

    alltop1acc = alltop3acc = alltop5acc = 0
    totalnum = 0
    alltopk = []
    ipdb.set_trace()
    while True:#这里不需要update-topk-sim-case，可以在case_corpus_imgretrieve里面改sim id file
        batch = data.next_batch()#keys:dict_keys(['caseid_in', 'rspvec_in', 'weighted_sim_imgvec_in'])
        batch_cnt += 1
        if batch is None:
            break

        # labels = model.get_similarity(batch, all_vecs)
        labels = model.get_similarity(batch, train_vecs)#batch中为相似样本的加权表征 train_vecs为img训练生成的特征
        #labels 32 5 \ batch \ train_vecs 3428, 128
        # labels = model.get_similarity (batch, test_vecs)

        # move from GPU to CPU
        topk_imgids_out = labels.cpu().data.tolist()# b, 5
        alltopk += topk_imgids_out
        caseids = batch['caseid_in'].tolist()


        batch_size = len(caseids)
        for b_id in range(batch_size):
            caseid = caseids[b_id][0]
            caseid = str(caseid)

            did, tid = test_caseid2didtid[caseid]
            gt_imgs = dialogues[did]['dialogue'][tid]['agent']['imgs']
            # gt_imgs = data['gt_rsp_img']
            all_valid_imgnames = []
            for gt_img in gt_imgs:
                gt_img2imgs = img2imgs[gt_img]
                all_valid_imgnames += gt_img2imgs
            all_valid_imgids = set()
            for iname in all_valid_imgnames:
                if iname in train_imgname2vecid:
                    validid = train_imgname2vecid[iname]
                    all_valid_imgids.add(validid)
                # if iname in test_imgname2vecid:
                #     validid = test_imgname2vecid[iname] + 3428
                #     all_valid_imgids.add(validid)


            totalnum += 1
            topk_imgids = topk_imgids_out[b_id]
#             ipdb.set_trace()
            top1acc = top3acc = top5acc = False
    
            for i, imgid in enumerate(topk_imgids):#检索到的相似样本
                if imgid in all_valid_imgids:#
                    if i == 0:
                        top1acc = top3acc = top5acc =True
                    elif i <= 2:
                        top3acc = top5acc =True
                    else:
                        top5acc = True

            if top1acc:
                alltop1acc += 1
            if top3acc:
                alltop3acc += 1
            if top5acc:
                alltop5acc += 1

    racall1 = alltop1acc * 1.0 / totalnum
    racall3 = alltop3acc * 1.0 / totalnum
    racall5 = alltop5acc * 1.0 / totalnum

    report = ""
    report += '{} Corpus recall@1: {:2.3f}%'.format(mode, racall1 * 100) + "\n"
    report += '{} Corpus recall@3: {:2.3f}%'.format(mode, racall3 * 100) + "\n"
    report += '{} Corpus recall@5: {:2.3f}%'.format(mode, racall5 * 100) + "\n"
    report += 'Total number of responses: %s ' % totalnum

    logger.info(report)

    json.dump(alltopk, dest_f, indent=4)

    return racall5
    
def generate_rsp(mode, model, data, config, evaluator, verbose=True, dest_f=None, weight_f = None, last_topk_file=None):
    model.eval()
    batch_cnt = 0

    data.epoch_init_pairs(config, shuffle=False, verbose=False, drop_last=False)
    logger.debug('Generation: {} batches for {} dataset'.format(data.num_batch_pairs, mode))
    ipdb.set_trace()
    while True:
        batch = data.next_batch_pairs()##这里不需要update-topk-sim-case，可以在corpus_genrsp里面改sim id file
        batch_cnt += 1
        if batch is None:
            break

        output, weight = model.get_rsp(batch)#  curr_rsp 被curr_rsp_encoder编码的结果；weight是curr_ctx 与sim_ctx的相似度
        # move from GPU to CPU
        output = output.cpu().data.numpy()
        for sample in output:
            sample_str = "\t".join(str(x) for x in sample)
            dest_f.write(sample_str + "\n")

        weight = weight.cpu().data.numpy()
        for sample in weight:
            sample_str = "\t".join(str(x) for x in sample)
            weight_f.write(sample_str + "\n")

    logger.debug('Generation curr_rsp_vec Done')
    
def compute_features(data, model, config):
    print('Computing features...')
    model.eval()
    features = []
    data.epoch_init_pairs(config, shuffle=False, verbose=False, drop_last=False)
    while True:#这里不需要update-topk-sim-case 因为不涉及sim case
        batch = data.next_batch_pairs()
        if batch is None:
            break


        with th.no_grad():
            case_out = model.get_case_out(batch)  # bs, dim_case_out 
            for sample in case_out:
                features.append(sample.tolist())  
#                 ipdb.set_trace()
    tensor = th.Tensor(features)
    return tensor.cpu()

# 计算对于不同类簇数目，所产生结果的评价指标（inertia，轮廓系数，卡林斯基哈拉巴斯指数（CHI））
def run_kmeans_sklearn(X, config, method=0,n_iter=10):
    '''
    :param X: 输入embedding矩阵，维度是 样本数量*关键词个数
    :param method: 选择聚类方法，0代表“kmeans”；1代表“kmeans++”
    :param n_iter:  执行次数，在多次执行中，选择inertia最小的
    :return: inertia_dic: 簇内平方和
              silhouette_dic: 轮廓系数
              calinski_harabasz_dic: CHI
    '''

    # 接收方法
    if method == 0:
        m = 'random'
        print('采用kmeans方法')
    elif method == 1:
        m = 'k-means++'
        print('采用kmeans++方法')

    # 设定要测试的类簇数目
    n_clusters = config.num_cluster
    cluster_result = {'im2cluster':[],'centroids':[]}
    # 存放 不同类簇数目 下的评价指标
    inertia_dic = {}
    silhouette_dic = {}
    calinski_harabasz_dic = {}
    
#     ipdb.set_trace()

    # 遍历所有要测试的类簇数目
    for i in n_clusters:
        print(f"k=={i}聚类中")
        result = KMeans(n_clusters=i, init=m, n_init=n_iter).fit(X)

        inertia = result.inertia_ #SSE
        labels = result.labels_ #所属簇
        centroids = result.cluster_centers_ #聚类中心表征


        # 添加inertia
        inertia_dic[i] = inertia
        # 添加silhouette
        s_score = silhouette_score(X, labels)
        silhouette_dic[i] = s_score
        # 添加CHI
        c_score = calinski_harabasz_score(X, labels)
        calinski_harabasz_dic[i] = c_score
        
        cluster_result['im2cluster'].append(labels)
        cluster_result['centroids'].append(centroids)
    

    # 排序，最优放最上面
    sorted_inertia_dic = sorted(inertia_dic.items(), key=lambda x: x[1])  # inertial越小越靠前
    sorted_silhouette_dic = sorted(silhouette_dic.items(), key=lambda x: x[1], reverse=True)  # 轮廓系数越大越靠前
    sorted_calinski_harabasz_dic = sorted(calinski_harabasz_dic.items(), key=lambda x: x[1], reverse=True)  # CHI越大越靠前


    print("kmeans聚类完成")
    return cluster_result, sorted_inertia_dic, sorted_silhouette_dic, sorted_calinski_harabasz_dic

def run_kmeans(x, config):
    """
    config:
        x: data to be clustered
    """
#     ipdb.set_trace()
    print('performing kmeans clustering')
    results = {'im2cluster':[],'centroids':[],'density':[], 'sorted_ids':[]}
    
    for seed, num_cluster in enumerate(config.num_cluster):
        # intialize faiss clustering parameters
        d = x.shape[1]  #BNC  此处N
        k = int(num_cluster)
        clus = faiss.Clustering(d, k)
        clus.verbose = True
        clus.niter = 20
        clus.nredo = 5
        clus.seed = seed
        clus.max_points_per_centroid = 1000
        clus.min_points_per_centroid = 10

        res = faiss.StandardGpuResources()
        cfg = faiss.GpuIndexFlatConfig()
        cfg.useFloat16 = False
        cfg.device = config.gpu    
        index = faiss.GpuIndexFlatL2(res, d, cfg)  

        clus.train(x, index)   

        D, I = index.search(x, 1) # for each sample, find cluster distance and assignments
        im2cluster = [int(n[0]) for n in I]
        
        # get cluster centroids
        centroids = faiss.vector_to_array(clus.centroids).reshape(k,d)
        
        
        
        # 取每个簇内， topk近距离的样本
        dic = [{} for c in range(k)]
        sorted_id = [[] for c in range(k)]         
        for im,i in enumerate(im2cluster):
            dic[i].update({im:D[im][0]})
        for i in range(k):
            sorted_dic = sorted(dic[i].items(),key=lambda s:s[1])#每个簇内根据距中心距离正序排序，返回元组
#             print(f"排序前：{dic[i]}")
#             print(f"排序后：{sorted_dic}")            
            sorted_id[i] = [tuple_sample[0] for tuple_sample in sorted_dic ] 
#             print(f"提取id：{sorted_id[i]}") 
        
        # sample-to-centroid distances for each cluster 
        Dcluster = [[] for c in range(k)] 
        for im,i in enumerate(im2cluster):
            Dcluster[i].append(D[im][0])
            
        
#         ipdb.set_trace()
        # concentration estimation (phi)        
        density = np.zeros(k)
        for i,dist in enumerate(Dcluster):
            if len(dist)>1:
                d = (np.asarray(dist)**0.5).mean()/np.log(len(dist)+10)            
                density[i] = d     
                
        #if cluster only has one point, use the max to estimate its concentration        
        dmax = density.max()
        for i,dist in enumerate(Dcluster):
            if len(dist)<=1:
                density[i] = dmax 

        density = density.clip(np.percentile(density,10),np.percentile(density,90)) #clamp extreme values for stability
        density = config.temperature*density/density.mean()  #scale the mean to temperature 
        
        # convert to cuda Tensors for broadcast
        centroids = th.Tensor(centroids).cuda()
        centroids = nn.functional.normalize(centroids, p=2, dim=1)  
        
        #保存sorted_id
        narry = np.zeros([len(sorted_id),len(max(sorted_id,key = lambda x: len(x)))])
        for i,j in enumerate(sorted_id):
            narry[i][0:len(j)] = j
        sorted_id_save = th.Tensor(narry).cuda()
#         print(sorted_id_save)

        im2cluster = th.LongTensor(im2cluster).cuda()               
        density = th.Tensor(density).cuda()
        
        results['centroids'].append(centroids)
        results['density'].append(density)
        results['im2cluster'].append(im2cluster)  
        results['sorted_ids'].append(sorted_id_save) 
        
    print("kmeans聚类完成")  
    return results

def save_vec(mode, flast_train_topk, flast_val_topk, flast_test_topk, model, config, data, vec_file, train_vecs=None, evaluator=None, train_data=None, topk_file=None, cluster_candidates_file=None):
    """
    vec_file: file to save the current generated vector results
    if train_data: encode train data, save in vec_file, and return train_vecs (list)
    if val_data or test data:
    encode and cosine with train_vecs in batch,
    save vecs and precision @ 1,3,5,10 in batch, save top 10 did tid
    return task_report, p1, p3, p5, p10
    """
    ipdb.set_trace()
    model.eval()
    batch_cnt = 0

    data.epoch_init_pairs(config, shuffle=False, verbose=False, drop_last=False)
    logger.debug('Generation: {} batches for {} dataset'.format(data.num_batch_pairs, mode))
    
    
    # for train data:
    if mode == "train":
        train_vecs = []
        while True:
            batch = data.next_batch_pairs()#update-topk-sim-case
            batch_cnt += 1
            if batch is None:
                break
                
            #update-topk-sim-case
            if config.ususe_rerank and not flast_train_topk is None:  
                batch = process_rerank(data, batch, flast_train_topk, config)                               
            case_out = model.get_case_out(batch)  # bs, dim_case_out
            # move from GPU to CPU
            case_out = case_out.cpu().data.numpy()
            for sample in case_out:
                train_vecs.append(sample.tolist())
                sample_str = "\t".join( str(x) for x in sample )
                vec_file.write(sample_str + "\n")
        return train_vecs


#     ipdb.set_trace()
    # for validation and test data:
    train_vecs = np.stack(train_vecs)
    all_caseids_in = []
    all_caseids_out = []

    num_cluster_candidates = None
    if cluster_candidates_file:
        num_cluster_candidates = config.num_cluster_candidates #50
    all_cluster_candidates = []
    

    while True:
        batch = data.next_batch_pairs()#update-topk-sim-case
        batch_cnt += 1
        if batch is None:
            break

        #update-topk-sim-case
        if config.ususe_rerank and not flast_train_topk is None: 
            if mode == "val":
                batch = process_rerank(train_data, batch, flast_val_topk, config)
            elif mode == "test":
                batch = process_rerank(train_data, batch, flast_test_topk, config)
            else:
                print(f"此处mode:{mode}应为val/test")
                
        case_out, topk_caseids_out, candidate_scores, candidate_caseids = model.get_similarity(batch, train_vecs, topk=(1,10), num_cluster_candidates=num_cluster_candidates)
        # move from GPU to CPU
        case_out = case_out.cpu().data.numpy()
        for sample in case_out:#写入当前数据子集的特征
            sample_str = "\t".join(str(x) for x in sample)
            vec_file.write(sample_str + "\n")

        if cluster_candidates_file:
            # [ [(id, score), ()], [], []   ] bs * 50 * 2
            candidate_caseids = candidate_caseids.cpu().data.tolist()  # bs, 50
            candidate_scores = candidate_scores.cpu().data.tolist()
            for caseids, scores in zip(candidate_caseids, candidate_scores):  # loop bs
                curr_candidates = []
                for caseid, score in zip(caseids, scores): # loop 50
                    curr_candidates.append((caseid, score))
                all_cluster_candidates+=curr_candidates

        topk_caseids_in = batch['topk_caseids_in'].tolist()
        topk_caseids_out = topk_caseids_out.cpu().data.tolist() # bs, K=10  前10个与给定case最相似的case的id
        all_caseids_in += topk_caseids_in
        all_caseids_out += topk_caseids_out
        

    levels = (1, 3, 5, 10)
    task_report, p1, p3, p5, p10 = evaluator.evaluate_all(mode, all_caseids_in, all_caseids_out, levels)

    if topk_file:
        json.dump(all_caseids_out, topk_file, indent=4)
    if cluster_candidates_file:
        json.dump(all_cluster_candidates, cluster_candidates_file, indent=4)

    logger.debug('Generation Done')
    logger.info(task_report)
    logger.debug('-' * 40)
    return p1, p3, p5, p10

def save_vec_train(mode, flast_train_topk, flast_val_topk, flast_test_topk, model, config, data, vec_file, train_vecs=None, evaluator=None, topk_file=None, cluster_candidates_file=None):
    model.eval()
    batch_cnt = 0

    data.epoch_init_pairs(config, shuffle=False, verbose=False, drop_last=False)
    logger.debug('Generation: {} batches for {} dataset'.format(data.num_batch_pairs, mode))
    
#     ipdb.set_trace()
    # for validation and test data:
    train_vecs = np.stack(train_vecs)
    all_caseids_in = []
    all_caseids_out = []

    num_cluster_candidates = None
    if cluster_candidates_file:
        num_cluster_candidates = config.num_cluster_candidates #50
    all_cluster_candidates = []
    

    while True:
        batch = data.next_batch_pairs()#update-topk-sim-case
        batch_cnt += 1
        if batch is None:
            break
            
            
        #update-topk-sim-case
        if config.ususe_rerank and not flast_train_topk is None: 
            if mode == "val":
                batch = process_rerank(train_data, batch, flast_val_topk, config)
            elif mode == "test":
                batch = process_rerank(train_data, batch, flast_test_topk, config)
                
            
        case_out, topk_caseids_out, candidate_scores, candidate_caseids = model.get_similarity(batch, train_vecs, topk=(1,10), num_cluster_candidates=num_cluster_candidates)

        if cluster_candidates_file:
            # [ [(id, score), ()], [], []   ] bs * 50 * 2
            candidate_caseids = candidate_caseids.cpu().data.tolist()  # bs, 50
            candidate_scores = candidate_scores.cpu().data.tolist()
            for caseids, scores in zip(candidate_caseids, candidate_scores):  # loop bs
                curr_candidates = []
                for caseid, score in zip(caseids, scores): # loop 50
                    curr_candidates.append((caseid, score))
                all_cluster_candidates+=curr_candidates[1:]

        topk_caseids_in = batch['topk_caseids_in'].tolist()
        topk_caseids_out = topk_caseids_out.cpu().data.tolist() # bs, K=10  前10个与给定case最相似的case的id
        all_caseids_in += topk_caseids_in
        for topk_caseids in topk_caseids_out:
            all_caseids_out.append(topk_caseids[1:])
        

    levels = (1, 3, 5, 10)
    task_report, p1, p3, p5, p10 = evaluator.evaluate_all(mode, all_caseids_in, all_caseids_out, levels)

    if topk_file:
        json.dump(all_caseids_out, topk_file, indent=4)
    if cluster_candidates_file:
        json.dump(all_cluster_candidates, cluster_candidates_file, indent=4)

    logger.debug('Generation Done')
    logger.info(task_report)
    logger.debug('-' * 40)
    return p1, p3, p5, p10



def gen_rsp_ctx_vec(model, train_data, val_data, test_data, config):
    save_path = 'dataset/ctx_rsp_feature_chen7200/'
    
    tb_path = os.path.join(config.saved_path, "tensorboard/")
    tb_logger = TBLogger(tb_path)

    mode_list = ['train', 'val', 'test']
    data = [train_data, val_data, test_data]
#     ipdb.set_trace()
    for i in range(len(mode_list)):
        ctx_vec_path = os.path.join(save_path, mode_list[i], 'ctx_vec.pt')
        ctx_tokenid_vec_path = os.path.join(save_path, mode_list[i], 'ctx_tokenid_vec.pt')
        ctx_tokenmask_vec_path = os.path.join(save_path, mode_list[i], 'ctx_tokenmask_vec.pt')
        rsp_vec_path = os.path.join(save_path, mode_list[i], 'rsp_vec.pt')
        rsp_tokenid_vec_path = os.path.join(save_path, mode_list[i], 'rsp_tokenid_vec.pt')
        rsp_tokenmask_vec_path = os.path.join(save_path, mode_list[i], 'rsp_tokenmask_vec.pt')
         
        
        save_ctx_rsp_vec(mode_list[i], model, ctx_vec_path, ctx_tokenid_vec_path, ctx_tokenmask_vec_path, rsp_vec_path, rsp_tokenid_vec_path, rsp_tokenmask_vec_path, data[i], config)

def validate(model, data, config, batch_cnt=None):
    model.eval()  # deactivate batch normalization and dropout
    data.epoch_init_pairs(config, shuffle=False, verbose=False, drop_last=False)
    losses = LossManager()
    while True:
        batch = data.next_batch_pairs()
        if batch is None:
            break
        loss = model(batch)
        losses.add_loss(loss)
        losses.add_backward_loss(model.model_sel_loss(loss, batch_cnt))  # a (weighted) sum of loss

    valid_loss = losses.avg_loss()  # mean of self.backward_losses
    logger.info(losses.pprint(data.mode))
    logger.info('--- Total loss = {}'.format(valid_loss))
    sys.stdout.flush()
    return valid_loss

def save_ctx_rsp_vec(mode, model, f_ctx, f_ctx_tokenid, f_ctx_tokenmask, f_rsp, f_rsp_tokenid, f_rsp_tokenmask, data, config):
    model.eval() 

    data.epoch_init_pairs(config, shuffle=False, verbose=False, drop_last=False)
   
    num_batch = data.num_batch_pairs
    batch_cnt = 0
    
    ctx_vec_save = []
    ctx_token_id = []
    ctx_token_mask = []
    rsp_vec_save = []
    rsp_token_id = []
    rsp_token_mask = []  
    while True:
        logger.info(f'batch_num: {mode}:{batch_cnt}/{num_batch}')
        # BATCH
        batch = data.next_batch_pairs()#batch-pairs 
        if batch is None:
            break
            
        batch_cnt += 1
        batch_size = batch.batch_size
#         ipdb.set_trace()
        with th.no_grad():
            ctx_vec, ctx_id_mask, rsp_vec ,rsp_id_mask= model.genvec(batch)
        ctx_vec_save.append(ctx_vec.cpu().data)
        ctx_token_id.append(ctx_id_mask['input_ids'].cpu().data)
        ctx_token_mask.append(ctx_id_mask['attention_mask'].cpu().data)
        rsp_vec_save.append(rsp_vec.cpu().data)
        rsp_token_id.append(rsp_id_mask['input_ids'].cpu().data)
        rsp_token_mask.append(rsp_id_mask['attention_mask'].cpu().data)
                                   
    ctx_vec_save = th.cat(ctx_vec_save)
    th.save(ctx_vec_save, f_ctx)
    ctx_token_id = th.cat(ctx_token_id)
    th.save(ctx_token_id, f_ctx_tokenid) 
    ctx_token_mask = th.cat(ctx_token_mask)
    th.save(ctx_token_mask, f_ctx_tokenmask) 
    rsp_vec_save = th.cat(rsp_vec_save)
    th.save(rsp_vec_save, f_rsp) 
    rsp_token_id = th.cat(rsp_token_id)
    th.save(rsp_token_id, f_rsp_tokenid) 
    rsp_token_mask = th.cat(rsp_token_mask)
    th.save(rsp_token_mask, f_rsp_tokenmask) 


def eval_text(mode, model, config, data, train_vecs=None, evaluator=None, dest_f=None):
    model.eval()
    batch_cnt = 0

    data.epoch_init(config, shuffle=False, verbose=False, drop_last=False)
    logger.debug('Generation: {} batches for {} dataset'.format(data.num_batch, mode))

    # for validation and test data:
    train_vecs = np.stack(train_vecs)
    all_caseids_in = []
    all_caseids_out = []

    while True:
        batch = data.next_batch()
        batch_cnt += 1
        if batch is None:
            break

        topk_caseids_out = model.get_similarity_text(batch, train_vecs, topk=(1,10))

        topk_caseids_in = batch['topk_caseids_in'].tolist()
        topk_caseids_out = topk_caseids_out.cpu().data.tolist() # bs, K=10
        all_caseids_in += topk_caseids_in
        all_caseids_out += topk_caseids_out

    levels = (1, 3, 5, 10)
    task_report, p1, p3, p5, p10 = evaluator.evaluate_all(mode, all_caseids_in, all_caseids_out, levels)

    if dest_f:
        json.dump(all_caseids_out, dest_f, indent=4)

    logger.debug('Generation Done')
    logger.info(task_report)
    logger.debug('-' * 40)
    return p1, p3, p5, p10

def eval_img(mode, model, config, data, train_vecs=None, evaluator=None):
    model.eval()
    batch_cnt = 0

    data.epoch_init(config, shuffle=False, verbose=False, drop_last=False)
    logger.debug('Generation: {} batches for {} dataset'.format(data.num_batch, mode))

    # for validation and test data:
    train_vecs = np.stack(train_vecs)
    all_caseids_in = []
    all_caseids_out = []

    while True:
        batch = data.next_batch()
        batch_cnt += 1
        if batch is None:
            break

        topk_caseids_out = model.get_similarity_img(batch, train_vecs, topk=(1,10))

        topk_caseids_in = batch['topk_caseids_in'].tolist()
        topk_caseids_out = topk_caseids_out.cpu().data.tolist() # bs, K=10
        all_caseids_in += topk_caseids_in
        all_caseids_out += topk_caseids_out

    levels = (1, 3, 5, 10)
    task_report, p1, p3, p5, p10 = evaluator.evaluate_all(mode, all_caseids_in, all_caseids_out, levels)

    logger.debug('Generation Done')
    logger.info(task_report)
    logger.debug('-' * 40)
    return p1, p3, p5, p10

def save_vec_50(mode, model, config, data, vec_file, train_vecs=None, evaluator=None, topk_file=None, cluster_candidates_file=None):
    """
    vec_file: file to save the current generated vector results
    if train_data: encode train data, save in vec_file, and return train_vecs (list)
    if val_data or test data:
    encode and cosine with train_vecs in batch,
    save vecs and precision @ 1,3,5,10 in batch, save top 10 did tid
    return task_report, p1, p3, p5, p10
    """
    model.eval()
    batch_cnt = 0

    data.epoch_init(config, shuffle=False, verbose=False, drop_last=False)
    logger.debug('Generation: {} batches for {} dataset'.format(data.num_batch, mode))

    # for train data:
    if mode == "train":
        train_vecs = []
        while True:
            batch = data.next_batch()
            batch_cnt += 1
            if batch is None:
                break
            case_out = model.get_case_out(batch)  # bs, dim_case_out
            # move from GPU to CPU
            case_out = case_out.cpu().data.numpy()
            for sample in case_out:
                train_vecs.append(sample.tolist())
                sample_str = "\t".join( str(x) for x in sample )
                vec_file.write(sample_str + "\n")
        return train_vecs

    if mode != 'val' and mode != 'test':
        logger.error('Invalid mode for vector generation. Use train, val, test.')
        return

    # for validation and test data:
    train_vecs = np.stack(train_vecs)
    all_caseids_in = []
    all_caseids_out = []

    num_cluster_candidates = None
    if cluster_candidates_file:
        num_cluster_candidates = config.num_cluster_candidates
    all_cluster_candidates = []

    while True:
        batch = data.next_batch()
        batch_cnt += 1
        if batch is None:
            break

        case_out, topk_caseids_out, candidate_scores, candidate_caseids = model.get_similarity(batch, train_vecs, topk=(1,50), num_cluster_candidates=num_cluster_candidates)
        # move from GPU to CPU
        case_out = case_out.cpu().data.numpy()
        for sample in case_out:
            sample_str = "\t".join(str(x) for x in sample)
            vec_file.write(sample_str + "\n")

        if cluster_candidates_file:
            # [ [(id, score), ()], [], []   ] bs * 50 * 2
            candidate_caseids = candidate_caseids.cpu().data.tolist()  # bs, 50
            candidate_scores = candidate_scores.cpu().data.tolist()
            for caseids, scores in zip(candidate_caseids, candidate_scores):  # loop bs
                curr_candidates = []
                for caseid, score in zip(caseids, scores): # loop 50
                    curr_candidates.append((caseid, score))
                all_cluster_candidates+=curr_candidates

        topk_caseids_in = batch['topk_caseids_in'].tolist()
        topk_caseids_out = topk_caseids_out.cpu().data.tolist() # bs, K=10
        all_caseids_in += topk_caseids_in
        all_caseids_out += topk_caseids_out

    levels = (1, 3, 5, 10)
    task_report, p1, p3, p5, p10 = evaluator.evaluate_all(mode, all_caseids_in, all_caseids_out, levels)

    if topk_file:
        json.dump(all_caseids_out, topk_file, indent=4)
    if cluster_candidates_file:
        json.dump(all_cluster_candidates, cluster_candidates_file, indent=4)

    logger.debug('Generation Done')
    logger.info(task_report)
    logger.debug('-' * 40)
    return p1, p3, p5, p10



