import os
import json
import argparse
from nltk.tokenize import word_tokenize
from reuse_crosscopy.utils import Pack
from reuse_crosscopy.evaluators import CaseEvaluator


def extract_venuename(info, gt_rsp):
    results = []
    if ':' in info:
        # slot, value = info.split(':')
        valuestart_idx = info.index(":")
        slot = info[:valuestart_idx]
        value = info[valuestart_idx+1:]
        if slot.strip() == 'venuename':
            venuename = value.strip().replace('\n', '').replace('\t', '')
            venuename = ' '.join(word_tokenize(venuename))
            if venuename in gt_rsp:
                results.append(venuename)
    return results

def get_request_venues(dialogue_acts, gt_rsp):
    # return provided venues in curr turn
    request_venues = []
    for info, act in dialogue_acts.items():
        venuename = extract_venuename(info, gt_rsp)
        request_venues += venuename
    return request_venues

def get_match(all_request_venues, all_pred_rsps):
    # return 1 if all_request_venues provided in pred rsps, 0 otherwise
    match = True
    for venue in all_request_venues:
        if venue not in all_pred_rsps:
            match = False
            break
    return 1 if match else 0

def extract_entity(info, act, gt_rsp):
    results = []
    if ':' in info:
        valuestart_idx = info.index(":")
        slot = info[:valuestart_idx]
        value = info[valuestart_idx+1:]
        if act.strip() != 'request':
            venuename = value.strip().replace('\n', '').replace('\t', '')
            venuename = ' '.join(word_tokenize(venuename))
            if venuename in gt_rsp:
                results.append(venuename)
    return results

def get_entities(dialogue_acts, gt_rsp):
    # get all act value in this turn
    entities = []
    for info, act in dialogue_acts.items():
        entity = extract_entity(info, act, gt_rsp)
        entities += entity
    return entities

def get_entityf1(gt_entities, pred_rsp):
    # return 1 if all gt_entities provided in pred rsps, 0 otherwise
    getentity = True
    for entity in gt_entities:
        if entity not in pred_rsp:
            getentity = False
            break
    return 1 if getentity else 0

def process(currresults, dialogues, test_ids, fout):
    currcaseid = -1
    # record for remaining set, remove set
    num_turn = 0 # number of turns that need to provide entity
    num_match = 0
    num_entitycorrect = 0
    for id in test_ids:
        turns = dialogues[id]['dialogue']
        all_request_venues = set()
        all_pred_rsps = ''
        for (turnid, turn) in enumerate(turns):
            if 'ctx' in turn:
                # load pred rsp
                currcaseid += 1
                pred_rsp = currresults[currcaseid * 3 + 1]
                start_idx = pred_rsp.index("Pred Rsp:") + len("Pred Rsp:")
                pred_rsp = pred_rsp[start_idx:].strip().replace('\n', '').replace('\t', '')
                pred_rsp = ' '.join(word_tokenize(pred_rsp))
                all_pred_rsps += ' ' + pred_rsp
                # load gt rsp from original dialogue data
                gt_rsp = currresults[currcaseid * 3]
                gt_start_idx = gt_rsp.index("True Rsp:") + len("True Rsp:")
                gt_rsp = gt_rsp[gt_start_idx:].strip().replace('\n', '').replace('\t', '')
                gt_rsp = ' '.join(word_tokenize(gt_rsp))
                # get dialogue act
                dialogue_acts = turn['agent']['dialog_act']
                # update dialogue level venues
                request_venues = get_request_venues(dialogue_acts, gt_rsp)
                all_request_venues.update(request_venues)
                # get turn level all entities
                gt_entities = get_entities(dialogue_acts, gt_rsp)
                if len(gt_entities) != 0:
                    num_turn += 1
                    num_entitycorrect += get_entityf1(gt_entities, pred_rsp) # 1 or 0
        num_match += get_match(all_request_venues, all_pred_rsps) # 1 or 0

    match_remain = num_match * 1.0 / (len(test_ids)-4)
    entityf1_remain = num_entitycorrect * 1.0 / num_turn
    print(f'number of tested dialogues: {(len(test_ids)-4)}')
    print('number of turns that need to provide entity: ' + str(num_turn))
    print(f'Match: {match_remain}\nEntity F1: {entityf1_remain}\n')

    fout.write(f'number of tested dialogues: {(len(test_ids) - 4)}\n')
    fout.write('number of turns that need to provide entity: ' + str(num_turn) + '\n')
    fout.write(f'Match: {match_remain}\nEntity F1: {entityf1_remain}\n\n')



def process_bleu(currresults, dialogues, test_ids, fout):
    currcaseid = -1
    # record for remaining set, remove set
    all_preds_remain = []
    all_labels_remain = []

    config = Pack({'K': 0})
    evaluator = CaseEvaluator(config)

    for id in test_ids:
        turns = dialogues[id]['dialogue']
        for (turnid, turn) in enumerate(turns):
            if 'ctx' in turn:
                # load pred rsp
                currcaseid += 1
                pred_rsp = currresults[currcaseid * 3 + 1]
                start_idx = pred_rsp.index("Pred Rsp:") + len("Pred Rsp:")
                pred_rsp = pred_rsp[start_idx:].strip().replace('\n', '').replace('\t', '')
                pred_rsp = ' '.join(word_tokenize(pred_rsp))
                # load gt rsp from original dialogue data
                gt_rsp = currresults[currcaseid * 3]
                gt_start_idx = gt_rsp.index("True Rsp:") + len("True Rsp:")
                gt_rsp = gt_rsp[gt_start_idx:].strip().replace('\n', '').replace('\t', '')
                gt_rsp = ' '.join(word_tokenize(gt_rsp))
                # update lists
                all_preds_remain.append([pred_rsp])
                all_labels_remain.append([gt_rsp])


    task_report_remain, bleu_remain = evaluator.evaluate('remain', all_preds_remain, all_labels_remain)
    print(task_report_remain)
    fout.write(task_report_remain + '\n')

def get_results(gen_result_path, dialogues, test_ids, fout):
    print('\nProcessing: ' + gen_result_path)
    fout.write('\nProcessing: ' + gen_result_path + '\n')
    currresults = open(gen_result_path, 'r').readlines()
    process(currresults, dialogues, test_ids, fout)

    process_bleu(currresults, dialogues, test_ids, fout)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--start_id', type=int, default=0)
    parser.add_argument('--end_id', type=int, default=49)
    parser.add_argument('--folder_path', type=str, default='outputs/pred_simctx_enc_fix_bs_k3_ctx5')
    parser.add_argument('--beam_size', type=int, default=-1)
    args = parser.parse_args()
    # get_results(args.gen_result)

    data_split_path = 'dataset' + '/data_split_random.json'
    data_split = json.load(open(data_split_path))
    test_ids = data_split['test']

    dialogue_path = 'dataset' + '/dialogues_with_ctx.json'
    dialogues = json.load(open(dialogue_path))

    fout = open(args.folder_path + "/eval_txt_results.log", "w")

    for model_id in range(args.start_id, args.end_id + 1):
        print('-----Model id : ' + str(model_id) + ' --------')
        fout.write('-----Model id : ' + str(model_id) + ' --------\n')
        if args.beam_size == -1:
            txt_path = args.folder_path + '/' + str(model_id) + '_test_file.txt'
        else:
            txt_path = args.folder_path + '/best_' + str(model_id) + '_test_file_beam' + str(args.beam_size) + '.txt'
        get_results(txt_path, dialogues, test_ids, fout)
        print()
        fout.write('\n')