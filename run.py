import torch, os, time, random, sys, json
import numpy as np
import logging
import torch.optim as optim
import torch.nn as nn

sys.path.append('./utils')
# from trans_module import TransitionModel, BertEncoder
from models import BERT_BiLSTM_CRF
from transformers import AdamW, get_linear_schedule_with_warmup
from collections import defaultdict
from torch.utils.data import DataLoader
import pandas as pd
from tqdm import tqdm
from fun import extract_arguments, get_arg_span, spans_to_tags

from config import get_config
config = get_config()
import datetime
now = datetime.datetime.now()
now_time_string = "{:0>4d}{:0>2d}{:0>2d}_{:0>2d}{:0>2d}{:0>2d}_{:0>5d}".format(
                now.year, now.month, now.day, now.hour, now.minute, now.second, config.seed)

os.environ['CUDA_VISIBLE_DEVICES'] = config.device
random.seed(config.seed)
np.random.seed(config.seed)
torch.manual_seed(config.seed)
torch.cuda.manual_seed(config.seed)
torch.cuda.manual_seed_all(config.seed)
os.environ['PYTHONHASHSEED'] = str(config.seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

save_path = config.save_path
save_path = os.path.join(save_path, now_time_string)
if not os.path.exists(save_path):
    os.makedirs(save_path)
else:
    print("save_path exists!!")
    exit(1)
with open(os.path.join(save_path, "config.json"), "w") as fp:
    json.dump(config.__dict__, fp)

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
formatter = logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s: - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
fh = logging.FileHandler(os.path.join(save_path, 'log.txt'))
fh.setLevel(logging.DEBUG)
fh.setFormatter(formatter)
# ch = logging.StreamHandler()
# ch.setLevel(logging.DEBUG)
# ch.setFormatter(formatter)
# logger.addHandler(ch) # output to terminal
logger.addHandler(fh) # output to file

tags2id = {'O': 0, 'B-Review': 1, 'I-Review': 2, 'E-Review': 3, 'S-Review': 4,
         'B-Reply': 1, 'I-Reply': 2, 'E-Reply': 3, 'S-Reply': 4,
         'B': 1, 'I': 2, 'E': 3, 'S': 4}

def load_data(file_path):
    sample_list = []
    with open(file_path, 'r') as fp:
        rr_pair_list = fp.read().split('\n\n\n')
        for rr_pair in rr_pair_list:
            if rr_pair == '':
                continue
            review, reply = rr_pair.split('\n\n')
            sample_review = {'sentences': [], 'bio_tags': [], 
                        'pair_tags': [], 'text_type': None, 'sub_ids': [], 'arg_spans': []}
            for line in review.strip().split('\n'):
                sent, bio_tag, pair_tag, text_type, sub_id = line.strip().split('\t')
                sample_review['sentences'].append(sent)
                sample_review['bio_tags'].append(bio_tag)
                sample_review['pair_tags'].append(pair_tag)
                sample_review['text_type'] = text_type
                sample_review['sub_ids'] = sub_id
            tags_ids = [tags2id[t] for t in sample_review['bio_tags']]
            sample_review['arg_spans'] = get_arg_span(tags_ids)

            sample_reply = {'sentences': [], 'bio_tags': [], 
                        'pair_tags': [], 'text_type': None, 'sub_ids': [], 'arg_spans': []}
            for line in reply.strip().split('\n'):
                sent, bio_tag, pair_tag, text_type, sub_id = line.strip().split('\t')
                sample_reply['sentences'].append(sent)
                sample_reply['bio_tags'].append(bio_tag)
                sample_reply['pair_tags'].append(pair_tag)
                sample_reply['text_type'] = text_type
                sample_reply['sub_ids'] = sub_id
            tags_ids = [tags2id[t] for t in sample_reply['bio_tags']]
            sample_reply['arg_spans'] = get_arg_span(tags_ids)

            rev_arg_2_rep_arg_dict = {}
            for rev_arg_span in sample_review['arg_spans']:
                rev_arg_pair_id = int(sample_review['pair_tags'][rev_arg_span[0]].split('-')[-1])
                rev_arg_2_rep_arg_dict[rev_arg_span] = []
                for rep_arg_span in sample_reply['arg_spans']:
                    rep_arg_pair_id = int(sample_reply['pair_tags'][rep_arg_span[0]].split('-')[-1])
                    if rev_arg_pair_id == rep_arg_pair_id:
                        rev_arg_2_rep_arg_dict[rev_arg_span].append(rep_arg_span)
            sample_review['rev_arg_2_rep_arg_dict'] = rev_arg_2_rep_arg_dict
            rep_seq_len = len(sample_reply['bio_tags'])
            rev_arg_2_rep_arg_tags_dict = {}
            for rev_arg_span, rep_arg_spans in rev_arg_2_rep_arg_dict.items():
                tags = spans_to_tags(rep_arg_spans, rep_seq_len)
                rev_arg_2_rep_arg_tags_dict[rev_arg_span] = tags
            sample_review['rev_arg_2_rep_arg_tags_dict'] = rev_arg_2_rep_arg_tags_dict

            rep_arg_2_rev_arg_dict = {}
            for rep_arg_span in sample_reply['arg_spans']:
                rep_arg_pair_id = int(sample_reply['pair_tags'][rep_arg_span[0]].split('-')[-1])
                rep_arg_2_rev_arg_dict[rep_arg_span] = []
                for rev_arg_span in sample_review['arg_spans']:
                    rev_arg_pair_id = int(sample_review['pair_tags'][rev_arg_span[0]].split('-')[-1])
                    if rep_arg_pair_id == rev_arg_pair_id:
                        rep_arg_2_rev_arg_dict[rep_arg_span].append(rev_arg_span)
            sample_reply['rep_arg_2_rev_arg_dict'] = rep_arg_2_rev_arg_dict
            rev_seq_len = len(sample_review['bio_tags'])
            rep_arg_2_rev_arg_tags_dict = {}
            for rep_arg_span, rev_arg_spans in rep_arg_2_rev_arg_dict.items():
                tags = spans_to_tags(rev_arg_spans, rev_seq_len)
                rep_arg_2_rev_arg_tags_dict[rep_arg_span] = tags
            sample_reply['rep_arg_2_rev_arg_tags_dict'] = rep_arg_2_rev_arg_tags_dict

            sample_list.append({'review': sample_review,
                                'reply': sample_reply})
    return sample_list

def args_metric(true_args_list, pred_args_list):
    tp, tn, fp, fn = 0, 0, 0, 0
    for true_args, pred_args in zip(true_args_list, pred_args_list):
        true_args_set = set(true_args)
        pred_args_set = set(pred_args)
        assert len(true_args_set) == len(true_args)
        assert len(pred_args_set) == len(pred_args)
        tp += len(true_args_set & pred_args_set)
        fp += len(pred_args_set - true_args_set)
        fn += len(true_args_set - pred_args_set)
    if tp + fp == 0:
        pre = tp/(tp + fp + 1e-10)
    else:
        pre = tp/(tp + fp)
    if tp + fn == 0:
        rec = tp/(tp + fn + 1e-10)
    else:
        rec = tp/(tp + fn)
    if pre == 0. and rec == 0.:
        f1 = (2 * pre *rec)/(pre + rec + 1e-10)
    else:
        f1 = (2 * pre *rec)/(pre + rec)
    acc = (tp + tn)/(tp + tn + fp +fn)
    return {'pre': pre, 'rec': rec, 'f1': f1, 'acc': acc}
    

def evaluate(model, data_list, graph_list):
    data_len = len(data_list)
    data_iter_len = (data_len // config.batch_size) + 1
    if data_len % config.batch_size == 0:
        data_iter_len -= 1
    model.eval()
    all_true_rev_args_list = []
    all_pred_rev_args_list = []
    all_true_rep_args_list = []
    all_pred_rep_args_list = []
    all_true_arg_pairs_list = []
    all_pred_arg_pairs_list = []
    threshold_pred_arg_pairs_list = []
    for batch_i in tqdm(range(data_iter_len)):
        data_batch = data_list[batch_i*config.batch_size:(batch_i+1)*config.batch_size]
        graph_batch = graph_list[batch_i*config.batch_size:(batch_i+1)*config.batch_size]

        review_para_tokens_list, review_tags_list = [], []
        reply_para_tokens_list, reply_tags_list = [], []
        rev_arg_2_rep_arg_tags_list = []
        true_arg_pairs_list = []
        for sample in data_batch:
            review_para_tokens_list.append(sample['review']['sentences'])
            tags_ids = [tags2id[tag] for tag in sample['review']['bio_tags']]
            review_tags_list.append(tags_ids)

            reply_para_tokens_list.append(sample['reply']['sentences'])
            tags_ids = [tags2id[tag] for tag in sample['reply']['bio_tags']]
            reply_tags_list.append(tags_ids)
            rev_arg_2_rep_arg_tags_list.append(sample['review']['rev_arg_2_rep_arg_tags_dict'])
            arg_pairs = []
            for rev_arg, rep_args in sample['review']['rev_arg_2_rep_arg_dict'].items():
                for rep_arg in rep_args:
                    arg_pairs.append((rev_arg, rep_arg))
            true_arg_pairs_list.append(arg_pairs)
        
        with torch.no_grad():
            pred_rev_args_list, pred_rep_args_list, pred_pair_args_list, pred_pair_args_2_list = \
                    model.predict(review_para_tokens_list, review_tags_list, rev_arg_2_rep_arg_tags_list, 
                        reply_para_tokens_list, reply_tags_list, graph_batch)
        
        true_rev_args_list = extract_arguments(review_tags_list)
        all_true_rev_args_list.extend(true_rev_args_list)
        all_pred_rev_args_list.extend(pred_rev_args_list)

        true_rep_args_list = extract_arguments(reply_tags_list)
        all_true_rep_args_list.extend(true_rep_args_list)
        all_pred_rep_args_list.extend(pred_rep_args_list)

        all_true_arg_pairs_list.extend(true_arg_pairs_list)

        pred_arg_pairs_list = []
        threshold_arg_pairs_list = []
        for pred_rep_args in pred_pair_args_list:
            pred_arg_pairs = []
            threshold_arg_pairs = {}
            for rev_arg, rep_args in pred_rep_args.items():
                for rep_arg, rep_arg_prob in zip(rep_args[0], rep_args[1]):
                    pred_arg_pairs.append((rev_arg, rep_arg))
                    threshold_arg_pairs[(rev_arg, rep_arg)] = rep_arg_prob
            pred_arg_pairs_list.append(pred_arg_pairs)
            threshold_arg_pairs_list.append(threshold_arg_pairs)

        pred_arg_pairs_2_list = []
        threshold_arg_pairs_2_list = []
        for pred_rep_args_2 in pred_pair_args_2_list:
            pred_arg_pairs = []
            threshold_arg_pairs = {}
            for rep_arg, rev_args in pred_rep_args_2.items():
                for rev_arg, rev_arg_prob in zip(rev_args[0], rev_args[1]):
                    pred_arg_pairs.append((rev_arg, rep_arg))
                    threshold_arg_pairs[(rev_arg, rep_arg)] = rev_arg_prob
            pred_arg_pairs_2_list.append(pred_arg_pairs)
            threshold_arg_pairs_2_list.append(threshold_arg_pairs) 

        for r_1_args, r_2_args in zip(threshold_arg_pairs_list, threshold_arg_pairs_2_list):
            pair_set = set(r_1_args.keys())&set(r_2_args.keys())
            for pair, p in r_1_args.items():
                if p > config.threshold:
                    pair_set.add(pair)
            for pair, p in r_2_args.items():
                if p > config.threshold:
                    pair_set.add(pair)
            threshold_pred_arg_pairs_list.append(list(pair_set))

        all_pred_arg_pairs_list.extend([list(set(a+b)) \
                 for a, b in zip(pred_arg_pairs_list, pred_arg_pairs_2_list)])

    res_dict = args_metric(all_true_rev_args_list, all_pred_rev_args_list)
    args_pair_dict = args_metric(all_true_arg_pairs_list, all_pred_arg_pairs_list)
    threshold_args_pair_dict = args_metric(all_true_arg_pairs_list, threshold_pred_arg_pairs_list)

    rep_dict = args_metric(all_true_rep_args_list, all_pred_rep_args_list)

    am_dict = args_metric(all_true_rev_args_list + all_true_rep_args_list, \
                    all_pred_rev_args_list + all_pred_rep_args_list)

    enhanced_all_pred_rep_args_list = []
    for idx, arg_pairs in enumerate(all_pred_arg_pairs_list):
        pred_rep_args = list(set([arg_pair[1] for arg_pair in arg_pairs]))
        enhanced_all_pred_rep_args_list.append(list(set(all_pred_rep_args_list[idx] + pred_rep_args)))

    enhanced_rep_dict = args_metric(all_true_rep_args_list, enhanced_all_pred_rep_args_list)
    enhanced_am_dict = args_metric(all_true_rev_args_list + all_true_rep_args_list, \
                        all_pred_rev_args_list + enhanced_all_pred_rep_args_list)
    return res_dict, rep_dict, am_dict, args_pair_dict, threshold_args_pair_dict, enhanced_rep_dict, enhanced_am_dict

    

    # loss = model(review_para_tokens_list, review_tags_list, rev_arg_2_rep_arg_tags_list, 
    #             reply_para_tokens_list, reply_tags_list)


train_list = load_data('data/processed_rr_sub/train.txt.bioes')
train_graph_list = json.load(open('data/processed_rr_sub/train.filtered.graph.json', 'r'))
dev_list = load_data('data/processed_rr_sub/dev.txt.bioes')
dev_graph_list = json.load(open('data/processed_rr_sub/dev.filtered.graph.json', 'r'))
test_list = load_data('data/processed_rr_sub/test.txt.bioes')
test_graph_list = json.load(open('data/processed_rr_sub/test.filtered.graph.json', 'r'))

train_len = len(train_list)
train_iter_len = (train_len // config.batch_size) + 1
if train_len % config.batch_size == 0:
    train_iter_len -= 1
num_training_steps = train_iter_len * config.epochs
num_warmup_steps = int(num_training_steps * config.warm_up)
logger.info('Data loaded.')

logger.info('Initializing model...')
model = BERT_BiLSTM_CRF(config)
model.cuda()
logger.info('Model initialized.')
bert_model_para = list(model.bert.parameters())
other_model_para = list(set(model.parameters())-set(bert_model_para))

optimizer_grouped_parameters = [
        {'params': [p for p in other_model_para if len(p.data.size()) > 1], 'weight_decay': config.weight_decay},
        {'params': [p for p in other_model_para if len(p.data.size()) == 1], 'weight_decay': 0.0},
        {'params': bert_model_para, 'lr': config.base_encoder_lr}
        ]
    
optimizer = AdamW(optimizer_grouped_parameters, config.finetune_lr)

total_batch, early_stop = 0, 0
best_batch, best_f1 = 0, 0.0
for epoch_i in range(config.epochs):
    logger.info("Running epoch: {}".format(epoch_i))
    loss_0, loss_1 = None, None
    last_loss_0, last_loss_1 = 0, 0
    bw_flag = False
    for batch_i in tqdm(range(train_iter_len)):
        model.train()
        train_batch = train_list[batch_i*config.batch_size:(batch_i+1)*config.batch_size]
        train_graph_batch = train_graph_list[batch_i*config.batch_size:(batch_i+1)*config.batch_size]
        if len(train_batch) <= 1:
            continue
        
        review_para_tokens_list, review_tags_list = [], []
        reply_para_tokens_list, reply_tags_list = [], []
        rev_arg_2_rep_arg_tags_list = []
        rep_arg_2_rev_arg_tags_list = []
        for sample in train_batch:
            review_para_tokens_list.append(sample['review']['sentences'])
            tags_ids = [tags2id[tag] for tag in sample['review']['bio_tags']]
            review_tags_list.append(tags_ids)

            reply_para_tokens_list.append(sample['reply']['sentences'])
            tags_ids = [tags2id[tag] for tag in sample['reply']['bio_tags']]
            reply_tags_list.append(tags_ids)

            rev_arg_2_rep_arg_tags_list.append(sample['review']['rev_arg_2_rep_arg_tags_dict'])
            rep_arg_2_rev_arg_tags_list.append(sample['reply']['rep_arg_2_rev_arg_tags_dict'])

        loss = model(review_para_tokens_list, review_tags_list, rev_arg_2_rep_arg_tags_list, rep_arg_2_rev_arg_tags_list,
                    reply_para_tokens_list, reply_tags_list, train_graph_batch)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
        optimizer.step()
        total_batch += 1

        if total_batch % config.showtime == 0:
            t_start = time.time()
            
            dev_rev_dict, dev_rep_dict, dev_am_dict, \
                dev_pair_res_dict, dev_threshold_args_pair_dict, enhanced_dev_rep_dict, enhanced_dev_am_dict \
                    = evaluate(model, dev_list, dev_graph_list)
            t_end = time.time()
            total_f1 = (dev_threshold_args_pair_dict['f1'] + dev_am_dict['f1']) / 2
            if total_f1  > best_f1:
                early_stop = 0
                best_f1 = total_f1                 
                torch.save(model.state_dict(), os.path.join(save_path, 'best_model.mdl'))
                logger.info('*'*20 +'best' + '*'*20)
                best_batch = total_batch 
                logger.info('*'*20 +'the performance in valid set...' + '*'*20)
                logger.info('running time: {}'.format(t_end - t_start))
                logger.info('total batch: {}'.format(total_batch))
                logger.info('rev ner f1:\t{:.4f}, pre:\t{:.4f}, rec:\t{:.4f}'.format(
                    dev_rev_dict['f1'], dev_rev_dict['pre'], dev_rev_dict['rec']))
                logger.info('rep ner f1:\t{:.4f}, pre:\t{:.4f}, rec:\t{:.4f}'.format(
                    dev_rep_dict['f1'], dev_rep_dict['pre'], dev_rep_dict['rec']))
                logger.info('am ner f1:\t{:.4f}, pre:\t{:.4f}, rec:\t{:.4f}'.format(
                    dev_am_dict['f1'], dev_am_dict['pre'], dev_am_dict['rec']))
                logger.info('pair f1:\t{:.4f}, pre:\t{:.4f}, rec:\t{:.4f}'.format(
                    dev_pair_res_dict['f1'], dev_pair_res_dict['pre'], dev_pair_res_dict['rec']))
                
                test_rev_dict, test_rep_dict, test_am_dict, \
                    test_pair_res_dict, test_threshold_args_pair_dict, enhanced_test_rep_dict, enhanced_test_am_dict \
                         = evaluate(model, test_list, test_graph_list)
                logger.info('*'*20 +'the performance in test set...' + '*'*20)
                logger.info('rev ner f1:\t{:.4f}, pre:\t{:.4f}, rec:\t{:.4f}'.format(
                    test_rev_dict['f1'], test_rev_dict['pre'], test_rev_dict['rec']))
                logger.info('rep ner f1:\t{:.4f}, pre:\t{:.4f}, rec:\t{:.4f}'.format(
                    test_rep_dict['f1'], test_rep_dict['pre'], test_rep_dict['rec']))
                logger.info('am ner f1:\t{:.4f}, pre:\t{:.4f}, rec:\t{:.4f}'.format(
                    test_am_dict['f1'], test_am_dict['pre'], test_am_dict['rec']))
                logger.info('pair f1:\t{:.4f}, pre:\t{:.4f}, rec:\t{:.4f}'.format(
                    test_pair_res_dict['f1'], test_pair_res_dict['pre'], test_pair_res_dict['rec']))
            


    early_stop += 1
    if early_stop > config.early_num or epoch_i == config.epochs-1:
        logger.info('early stop:' + str(early_stop))
        break
