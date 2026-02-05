from pprg_end2end.evaluators import BaseEvaluator, BLEUScorer
import ipdb
class CaseEvaluator(BaseEvaluator):
    def __init__(self, config):
        self.K = config.K

    def evaluate(self, mode, all_preds, all_labels):
        report = "Evaluate " + mode + " data :\n"

        assert len(all_preds) == len(all_labels)
        total = len(all_preds)
        bleuscorer = BLEUScorer()
        bleu_score = bleuscorer.score(all_preds, all_labels)

        report = ""
        report += '{} Corpus BLEU: {:2.3f}%'.format(mode, bleu_score * 100) + "\n"
        report += 'Total number of responses: %s ' % total

        return report, bleu_score


class CaseEvaluator_retrieval(BaseEvaluator):
    def __init__(self, config):
        self.K = config.K

    def evaluate_all(self, mode, all_caseids_in, all_caseids_out, levels=(1,3,5,10)):
        """
        consider top10 results precision, and top1 paired case coverage
        all_caseids_in: testset_size * K, [[1,2,3], [4,5,7]]
        all_caseids_out: testset_size * K, [[1,2,3], [4,5,6]]
        """
        assert len(all_caseids_in) == len(all_caseids_out)
        total = len(all_caseids_in)
        K = len(all_caseids_in[0])

        results = dict()
        for level in levels:
            results[level] = {'tp10': 0,'tp3': 0,'top1_covered': 0, 'top3_covered': 0}

        for gt_ids, pred_ids in zip(all_caseids_in, all_caseids_out):
            top1_gt_ids = gt_ids[0]
            top3_gt_ids = gt_ids[:3]
#             ipdb.set_trace()
            for k in levels:
                topk_pred_ids = pred_ids[:k] #预测的前k结果
                tp10 = len([t for t in topk_pred_ids if t in gt_ids])#预测的前k结果，gt的前10个结果，这俩匹配的数量
                results[k]['tp10'] += tp10
                tp3 = len([t for t in topk_pred_ids if t in top3_gt_ids])#预测的前k结果，gt的前3个结果，这俩匹配的数量
                results[k]['tp3'] += tp3
                if top1_gt_ids in topk_pred_ids:#预测的前k结果，gt的前1个结果，这俩匹配的数量
                    results[k]['top1_covered'] += 1
                if tp3 == 3:
                    results[k]['top3_covered'] += 1
#                 recall_case = len([t for t in topk_pred_ids if t in top3_gt_ids])    
#                 results[k]['top3_covered'] += 1
             

        # get precision, recall, f1, accuracy
        report = "Evaluate " + mode + " data :\n"
#         ipdb.set_trace()
        for k in levels:
            tp10, tp3, top1_covered, top3_covered = results[k]['tp10'], results[k]['tp3'], results[k]['top1_covered'], results[k]['top3_covered']
            prec10 = tp10 / (total * k + 10e-20)
            prec3 = tp3 / (total * k + 10e-20)
            top1coverage = top1_covered / (total + 10e-20)
            top3coverage = top3_covered / (total + 10e-20)
            results[k]['p10'], results[k]['p3'], results[k]['top1coverage'], results[k]['top3coverage'] = prec10 * 100, prec3 * 100, top1coverage * 100, top3coverage * 100
            report += '@{}: top10 prec: {:2.2f}%, top3 prec: {:2.2f}%, top1 coverage: {:2.2f}%, top3 coverage: {:2.2f}%'.format(k, results[k]['p10'], results[k]['p3'], results[k]['top1coverage'], results[k]['top3coverage']) + "\n"
        report += 'Total number of cases: %s ' % total
#         ipdb.set_trace()
#         tp10：预测的前10个案例ID中有至少一个案例ID出现在真实案例ID中，那么这个测试案例就被认为是一个tp10的案例
#         tp3：预测的前3个案例ID中有至少一个案例ID出现在真实案例ID中。
#         top1coverage：在前k个预测案例ID中，真实案例的第一个案例ID是否被预测到。
#         top3coverage：在前k个预测案例ID中，真实案例的前3个案例ID是否都被预测到。

        return report, results[1]['top3coverage'], results[3]['top3coverage'], results[5]['top3coverage'], results[10]['top3coverage']

    def evaluate(self, mode, all_caseids_in, all_caseids_out, levels=(1,3,5,10)):
        """
        only consider top1 paired case coverage
        all_caseids_in: testset_size * K, [[1,2,3], [4,5,7]],
        all_caseids_out: testset_size * K, [[1,2,3], [4,5,6]]
        """
        report = "Evaluate " + mode + " data :\n"

        assert len(all_caseids_in) == len(all_caseids_out)
        total = len(all_caseids_in)
        results = dict()
        for level in levels:
            results[level] = {'top1_covered': 0}

        for gt_ids, pred_ids in zip(all_caseids_in, all_caseids_out):
            top1_gt_ids = gt_ids[0]
            for k in levels:
                topk_pred_ids = pred_ids[:k]
                if top1_gt_ids in topk_pred_ids:
                    results[k]['top1_covered'] += 1

        for k in levels:
            top1_covered = results[k]['top1_covered']
            accuracy = top1_covered / (total + 10e-20)
            results[k]['top1coverage'] = accuracy * 100
            report += 'acc@{}: {:2.2f}%'.format(k, results[k]['top1coverage']) + "\n"
        report += 'Total number of cases: %s ' % total

        return report, results[1]['top1coverage'], results[3]['top1coverage'], results[5]['top1coverage'], results[10]['top1coverage']
