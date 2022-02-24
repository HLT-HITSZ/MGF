import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import sys
sys.path.append('./utils')
from transformers import BertModel, BertTokenizer
from copy import deepcopy
import numpy as np
from copy import deepcopy
from torch.nn.utils.rnn import pack_sequence, pad_packed_sequence, pad_sequence
from torchcrf import CRF
from fun import extract_arguments, get_arg_span, spans_to_tags
import dgl
from dgl.nn import GraphConv, EdgeWeightNorm
import torch

class BERT_BiLSTM_CRF(nn.Module):
    
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.mlp_size = config.mlp_size
        self.dropout = config.dropout
        self.scale_factor = config.scale_factor
        self.am_weight = config.am_weight

        self.bert = BertModel.from_pretrained(config.bert_path)
        self.special_tokens = ['[ENDL]', '[TAB]', '[LINE]',
                                '[EQU]', '[URL]', '[NUM]',
                                '[SPE]']
        self.tokenizer = BertTokenizer.from_pretrained(config.bert_path,
                        additional_special_tokens=self.special_tokens)

        self.am_bilstm = nn.LSTM(config.bert_output_size, config.hidden_size, \
                        num_layers=1, bidirectional=True, batch_first=True)
        
        self.pair_bilstm = nn.LSTM(config.hidden_size * 4, config.hidden_size, \
                num_layers=1, bidirectional=True, batch_first=True)
        self.pair_2_bilstm = nn.LSTM(config.hidden_size * 4, config.hidden_size, \
                num_layers=1, bidirectional=True, batch_first=True)
        
        self.am_hidden2tag = nn.Linear(config.hidden_size*2, config.num_tags)
        self.pair_hidden2tag = nn.Linear(config.hidden_size*2, config.num_tags)
        self.pair_2_hidden2tag = nn.Linear(config.hidden_size*2, config.num_tags)
        self.am_crf = CRF(config.num_tags, batch_first=True)
        self.pair_crf = CRF(config.num_tags, batch_first=True)
        self.pair_2_crf = CRF(config.num_tags, batch_first=True)
        self.dropout = nn.Dropout(p=config.dropout)

        self.conv = GraphConv(self.hidden_size*2, self.hidden_size*2, norm='both', weight=True, bias=True)
        self.norm = EdgeWeightNorm(norm='both')
    
    def bert_emb(self, para_tokens_list):
        para_len_list = [len(para) for para in para_tokens_list]
        max_para_len = max(para_len_list)

        sent_tokens_list = [sent for para in para_tokens_list for sent in para]

        tokenized_sent = self.tokenizer(sent_tokens_list, padding=True)
        sent_ids = [self.tokenizer.convert_tokens_to_ids(sent.split(' ')) 
                    for sent in sent_tokens_list]
        ids_padding_list, mask_list = self.padding_and_mask(sent_ids)
        ids_padding_tensor = torch.tensor(ids_padding_list).cuda()
        mask_tensor = torch.tensor(mask_list).cuda()
        bert_outputs = self.bert(ids_padding_tensor, attention_mask = mask_tensor)
        last_hidden_state = bert_outputs.last_hidden_state[:, 1:-1, :]
        sent_emb = last_hidden_state.mean(dim=-2)
        # sent_emb = last_hidden_state.max(dim=-2)[0]
        # sent_emb = bert_outputs.pooler_output
        sent_emb = self.dropout(sent_emb)
        return sent_emb

    def am_tagging(self, para_tokens_list, tags_list, h=None, c=None, mode='train'):
        sent_num_list = [len(para) for para in para_tokens_list]
        max_sent_num = max(sent_num_list) 
        tags_list, tags_mask = self.padding_and_mask(tags_list)
        tags_tensor = torch.tensor(tags_list).cuda()
        tags_mask_tensor = torch.tensor(tags_mask).cuda()
        sent_emb = self.bert_emb(para_tokens_list)

        para_emb = torch.split(sent_emb, sent_num_list, 0)
        para_emb_packed = pack_sequence(para_emb, enforce_sorted=False)
        if h != None and c != None:
            para_lstm_out_packed, (h, c) = self.am_bilstm(para_emb_packed, (h, c))
        else:
            para_lstm_out_packed, (h, c) = self.am_bilstm(para_emb_packed)
        para_lstm_out_padded, _ = pad_packed_sequence(para_lstm_out_packed, batch_first=True)
        para_lstm_out = para_lstm_out_padded[tags_mask_tensor.bool()]
        tags_prob = self.am_hidden2tag(para_lstm_out)
        tags_prob_list = torch.split(tags_prob, sent_num_list, 0)
        tags_prob_padded = pad_sequence(tags_prob_list, batch_first=True)
        
        loss, pred_args = None, None
        if mode == 'train':
            loss = -1*self.am_crf(tags_prob_padded, tags_tensor, mask=tags_mask_tensor.byte())
        else:
            pred_args = self.am_crf.decode(tags_prob_padded, tags_mask_tensor.byte())

        para_lstm_out_list = []
        for batch_i, sent_num in enumerate(sent_num_list):
            para_lstm_out_list.append(para_lstm_out_padded[batch_i][:sent_num])
        
        return loss, pred_args, para_lstm_out_list, h, c
    
    # def GCN_layer(self, ):

    
    def forward(self, review_para_tokens_list, review_tags_list, rev_arg_2_rep_arg_tags_list, rep_arg_2_rev_arg_tags_list,
                    reply_para_tokens_list, reply_tags_list, train_graph_list):
        rev_len_list = [len(_) for _ in review_para_tokens_list]
        rep_len_list = [len(_) for _ in reply_para_tokens_list]
        # review
        rev_loss, _, rev_para_lstm_out_list, h_rev, c_rev \
            = self.am_tagging(review_para_tokens_list, review_tags_list, mode='train')

        # reply
        rep_loss, _, rep_para_lstm_out_list, _, _ \
            = self.am_tagging(reply_para_tokens_list, reply_tags_list, mode='train')

        # GCN
        rev_para_gcn_out_list = []
        rep_para_gcn_out_list = []
        for batch_i in range(len(train_graph_list)):
            graph = train_graph_list[batch_i]
            s, e, w = zip(*graph)
            g = dgl.graph((torch.tensor(s).cuda(), torch.tensor(e).cuda()))
            edge_weight = torch.tensor(w).to(torch.float32).cuda()
            norm_edge_weight = self.norm(g, edge_weight)
            # g = dgl.add_self_loop(g)
            sent_feat = torch.cat([rev_para_lstm_out_list[batch_i], \
                                    rep_para_lstm_out_list[batch_i]])
            gcn_sent_feat = F.relu(self.conv(g, sent_feat, edge_weight=norm_edge_weight))
            rev_sent_len = rev_para_lstm_out_list[batch_i].shape[0]
            rev_para_gcn_out_list.append(gcn_sent_feat[:rev_sent_len])
            rep_para_gcn_out_list.append(gcn_sent_feat[rev_sent_len:])

        # pair
        review_args_rep = []
        pair_tags_list = []
        rev_args_span_list = []
        for batch_i, rev_arg_2_rep_arg_tags in enumerate(rev_arg_2_rep_arg_tags_list):
            rev_args_span = []
            for rev_arg_span, rep_arg_tags in rev_arg_2_rep_arg_tags.items():
                review_args_rep.append(rev_para_gcn_out_list[batch_i][rev_arg_span[0]:rev_arg_span[1]+1].mean(dim=-2))
                pair_tags_list.append(rep_arg_tags)
                rev_args_span.append(rev_arg_span)
            rev_args_span_list.append(rev_args_span)

        review_args_rep_tensor = torch.stack(review_args_rep)
        num_args_list = [len(d) for d in rev_arg_2_rep_arg_tags_list]
        review_args_rep_list = torch.split(review_args_rep_tensor, num_args_list, 0)
        
        num_sent_list = []
        num_args_list = []
        rep_with_rev_rep_list = []
        for batch_i, review_args_rep in enumerate(review_args_rep_list):
            num_args_list.append(review_args_rep.shape[0])
            for arg_idx in range(review_args_rep.shape[0]):
                num_sent = rep_para_gcn_out_list[batch_i].shape[0]
                num_sent_list.append(num_sent)
                args_rep = review_args_rep[arg_idx].unsqueeze(0).repeat(num_sent, 1)
                rep_with_rev_rep = torch.cat([rep_para_gcn_out_list[batch_i], args_rep], dim=-1)
                rep_with_rev_rep_list.append(rep_with_rev_rep)
            
        rep_with_rev_rep_packed = pack_sequence(rep_with_rev_rep_list, enforce_sorted=False)
        rep_with_rev_lstm_out_packed, _ = self.pair_bilstm(rep_with_rev_rep_packed)
        rep_with_rev_lstm_out_padded, _ = pad_packed_sequence(rep_with_rev_lstm_out_packed, batch_first=True)    
 
        rep_tags_len_list = [len(tags) for tags in pair_tags_list]
        rep_max_tags_len = max(rep_tags_len_list)
        pair_tags_list, tags_mask = self.padding_and_mask(pair_tags_list)
        pair_tags_tensor = torch.tensor(pair_tags_list).cuda()
        pair_tags_mask_tensor = torch.tensor(tags_mask).cuda() 

        rep_with_rev_lstm_out = rep_with_rev_lstm_out_padded[pair_tags_mask_tensor.bool()]
        pair_tags_prob = self.pair_hidden2tag(rep_with_rev_lstm_out)
        pair_tags_prob_list = torch.split(pair_tags_prob, num_sent_list, 0)
        pair_tags_prob_padded = pad_sequence(pair_tags_prob_list, batch_first=True)  

        pair_loss = -1*self.pair_crf(pair_tags_prob_padded, pair_tags_tensor, mask=pair_tags_mask_tensor.byte())
    


        reply_args_rep = []
        pair_tags_2_list = []
        rep_args_span_list = []
        for batch_i, rep_arg_2_rev_arg_tags in enumerate(rep_arg_2_rev_arg_tags_list):
            rep_args_span = []
            for rep_arg_span, rev_arg_tags in rep_arg_2_rev_arg_tags.items():
                reply_args_rep.append(rep_para_gcn_out_list[batch_i][rep_arg_span[0]:rep_arg_span[1]+1].mean(dim=-2))
                pair_tags_2_list.append(rev_arg_tags)
                rep_args_span.append(rep_arg_span)
            rep_args_span_list.append(rep_args_span)

        reply_args_rep_tensor = torch.stack(reply_args_rep)
        num_args_list = [len(d) for d in rep_arg_2_rev_arg_tags_list]
        reply_args_rep_list = torch.split(reply_args_rep_tensor, num_args_list, 0)
        
        num_sent_list = []
        num_args_list = []
        rev_with_rep_rep_list = []
        for batch_i, reply_args_rep in enumerate(reply_args_rep_list):
            num_args_list.append(reply_args_rep.shape[0])
            for arg_idx in range(reply_args_rep.shape[0]):
                num_sent = rev_para_gcn_out_list[batch_i].shape[0]
                num_sent_list.append(num_sent)
                args_rep = reply_args_rep[arg_idx].unsqueeze(0).repeat(num_sent, 1)
                rev_with_rep_rep = torch.cat([rev_para_gcn_out_list[batch_i], args_rep], dim=-1)
                rev_with_rep_rep_list.append(rev_with_rep_rep)
            
        rev_with_rep_rep_packed = pack_sequence(rev_with_rep_rep_list, enforce_sorted=False)
        rev_with_rep_lstm_out_packed, _ = self.pair_2_bilstm(rev_with_rep_rep_packed)
        rev_with_rep_lstm_out_padded, _ = pad_packed_sequence(rev_with_rep_lstm_out_packed, batch_first=True)    
 
        rev_tags_len_list = [len(tags) for tags in pair_tags_2_list]
        rev_max_tags_len = max(rev_tags_len_list)
        pair_tags_2_list, tags_mask = self.padding_and_mask(pair_tags_2_list)
        pair_tags_2_tensor = torch.tensor(pair_tags_2_list).cuda()
        pair_tags_mask_tensor = torch.tensor(tags_mask).cuda() 

        rev_with_rep_lstm_out = rev_with_rep_lstm_out_padded[pair_tags_mask_tensor.bool()]
        pair_tags_2_prob = self.pair_2_hidden2tag(rev_with_rep_lstm_out)
        pair_tags_2_prob_list = torch.split(pair_tags_2_prob, num_sent_list, 0)
        pair_tags_2_prob_padded = pad_sequence(pair_tags_2_prob_list, batch_first=True)  

        pair_2_loss = -1*self.pair_2_crf(pair_tags_2_prob_padded, pair_tags_2_tensor, mask=pair_tags_mask_tensor.byte())

        return self.am_weight*(rev_loss + rep_loss) + (1.0-self.am_weight)*(pair_loss + pair_2_loss)
    
    def predict(self, review_para_tokens_list, review_tags_list, rev_arg_2_rep_arg_tags_list,
                    reply_para_tokens_list, reply_tags_list, graph_list):
        # review
        _, pred_rev_args, rev_para_lstm_out_list, h_rev, c_rev \
            = self.am_tagging(review_para_tokens_list, review_tags_list, mode='pred')
        pred_rev_args_list = extract_arguments(pred_rev_args)

        # reply
        _, pred_rep_args, rep_para_lstm_out_list, _, _ \
            = self.am_tagging(reply_para_tokens_list, reply_tags_list, mode='pred')
        pred_rep_args_list = extract_arguments(pred_rep_args)

        # GCN
        rev_para_gcn_out_list = []
        rep_para_gcn_out_list = []
        for batch_i in range(len(graph_list)):
            graph = graph_list[batch_i]
            s, e, w = zip(*graph)
            g = dgl.graph((torch.tensor(s).cuda(), torch.tensor(e).cuda()))
            edge_weight = torch.tensor(w).to(torch.float32).cuda()
            norm_edge_weight = self.norm(g, edge_weight)
            # g = dgl.add_self_loop(g)
            sent_feat = torch.cat([rev_para_lstm_out_list[batch_i], \
                                    rep_para_lstm_out_list[batch_i]])
            gcn_sent_feat = F.relu(self.conv(g, sent_feat, edge_weight=norm_edge_weight))
            rev_sent_len = rev_para_lstm_out_list[batch_i].shape[0]
            rev_para_gcn_out_list.append(gcn_sent_feat[:rev_sent_len])
            rep_para_gcn_out_list.append(gcn_sent_feat[rev_sent_len:])

        # pair
        pred_args_pair_dict_list = []
        review_args_rep = []
        for batch_i, pred_arguments in enumerate(pred_rev_args_list):
            for rev_arg_span in pred_arguments:
                review_args_rep.append(rev_para_gcn_out_list[batch_i][rev_arg_span[0]:rev_arg_span[1]+1].mean(dim=-2))

        if review_args_rep == []:
            pred_args_pair_dict_list = [{} for t in pred_rev_args_list]
        else:
            review_args_rep_tensor = torch.stack(review_args_rep)
            num_args_list = [len(d) for d in pred_rev_args_list]
            review_args_rep_list = torch.split(review_args_rep_tensor, num_args_list, 0)

            num_sent_list = []
            rev_with_rep_rep_list = []
            for batch_i, review_args_rep in enumerate(review_args_rep_list):
                for arg_idx in range(review_args_rep.shape[0]):
                    num_sent = rep_para_gcn_out_list[batch_i].shape[0]
                    num_sent_list.append(num_sent)
                    args_rep = review_args_rep[arg_idx].unsqueeze(0).repeat(num_sent, 1)
                    rev_with_rep_rep = torch.cat([rep_para_gcn_out_list[batch_i], args_rep], dim=-1)
                    rev_with_rep_rep_list.append(rev_with_rep_rep)
                
            rev_with_rep_rep_packed = pack_sequence(rev_with_rep_rep_list, enforce_sorted=False)
            rev_with_rep_lstm_out_packed, _ = self.pair_bilstm(rev_with_rep_rep_packed)
            rev_with_rep_lstm_out_padded, rep_len_list = pad_packed_sequence(rev_with_rep_lstm_out_packed, batch_first=True)    

            max_rep_len = max(rep_len_list)
            tags_mask = [[1]*rep_len + [0]*(max_rep_len-rep_len) for rep_len in rep_len_list]
            pair_tags_mask_tensor = torch.tensor(tags_mask).cuda()

            rev_with_rep_lstm_out = rev_with_rep_lstm_out_padded[pair_tags_mask_tensor.bool()]
            pair_tags_prob = self.pair_hidden2tag(rev_with_rep_lstm_out)
            pair_tags_prob_list = torch.split(pair_tags_prob, num_sent_list, 0)
            pair_tags_prob_padded = pad_sequence(pair_tags_prob_list, batch_first=True)  

            pred_pair_rep_args_tag = self.pair_crf.decode(pair_tags_prob_padded, pair_tags_mask_tensor.byte())
            pred_pair_rep_args_list = extract_arguments(pred_pair_rep_args_tag)

            # prob
            pred_pair_rep_args_prob_list = []
            for idx in range(pair_tags_prob_padded.shape[0]):
                full_emission = pair_tags_prob_padded[idx]
                full_mask = pair_tags_mask_tensor[idx].byte()
                full_tags = torch.tensor(pred_pair_rep_args_tag[idx]).cuda()
                pred_pair_rep_args_probs = []
                for pred_span in pred_pair_rep_args_list[idx]:
                    start, end = pred_span
                    emission = full_emission[start: end+1].unsqueeze(1)  # [1, end-start+1]
                    mask = full_mask[start: end+1].unsqueeze(1)    # [1, end-start+1]
                    tags = full_tags[start:end+1].unsqueeze(1)  # [1, end-start+1]
                    numerator = self.pair_crf._compute_score(emission, tags, mask)
                    denominatr = self.pair_crf._compute_normalizer(emission, mask)
                    llh = numerator - denominatr
                    prob = llh.exp()
                    pred_pair_rep_args_probs.append(prob.tolist()[0])
                pred_pair_rep_args_prob_list.append(pred_pair_rep_args_probs)

            i = 0
            for pred_arguments in pred_rev_args_list:
                pred_args_pair_dict = {}
                for args in pred_arguments:
                    pred_args_pair_dict[args] = (pred_pair_rep_args_list[i], \
                                                    pred_pair_rep_args_prob_list[i])
                    i+=1
                pred_args_pair_dict_list.append(pred_args_pair_dict)



        pred_args_pair_2_dict_list = []
        reply_args_rep = []
        for batch_i, pred_arguments in enumerate(pred_rep_args_list):
            for rep_arg_span in pred_arguments:
                reply_args_rep.append(rep_para_gcn_out_list[batch_i][rep_arg_span[0]:rep_arg_span[1]+1].mean(dim=-2))

        if reply_args_rep == []:
            pred_args_pair_2_dict_list = [{} for t in pred_rep_args_list]
        else:
            reply_args_rep_tensor = torch.stack(reply_args_rep)
            num_args_list = [len(d) for d in pred_rep_args_list]
            reply_args_rep_list = torch.split(reply_args_rep_tensor, num_args_list, 0)
            
            num_sent_list = []
            rev_with_rep_rep_list = []
            for batch_i, reply_args_rep in enumerate(reply_args_rep_list):
                for arg_idx in range(reply_args_rep.shape[0]):
                    num_sent = rev_para_gcn_out_list[batch_i].shape[0]
                    num_sent_list.append(num_sent)
                    args_rep = reply_args_rep[arg_idx].unsqueeze(0).repeat(num_sent, 1)
                    rev_with_rep_rep = torch.cat([rev_para_gcn_out_list[batch_i], args_rep], dim=-1)
                    rev_with_rep_rep_list.append(rev_with_rep_rep)
                
            rev_with_rep_rep_packed = pack_sequence(rev_with_rep_rep_list, enforce_sorted=False)
            rev_with_rep_lstm_out_packed, _ = self.pair_2_bilstm(rev_with_rep_rep_packed)
            rev_with_rep_lstm_out_padded, rev_len_list = pad_packed_sequence(rev_with_rep_lstm_out_packed, batch_first=True)    

            max_rev_len = max(rev_len_list)
            tags_mask = [[1]*rev_len + [0]*(max_rev_len-rev_len) for rev_len in rev_len_list]
            pair_tags_mask_tensor = torch.tensor(tags_mask).cuda()

            rev_with_rep_lstm_out = rev_with_rep_lstm_out_padded[pair_tags_mask_tensor.bool()]
            pair_tags_2_prob = self.pair_2_hidden2tag(rev_with_rep_lstm_out)
            pair_tags_2_prob_list = torch.split(pair_tags_2_prob, num_sent_list, 0)
            pair_tags_2_prob_padded = pad_sequence(pair_tags_2_prob_list, batch_first=True)  

            pred_pair_rep_args_tag_2 = self.pair_2_crf.decode(pair_tags_2_prob_padded, pair_tags_mask_tensor.byte())
            pred_pair_rep_args_2_list = extract_arguments(pred_pair_rep_args_tag_2)

            # prob
            pred_pair_rep_args_prob_2_list = []
            for idx in range(pair_tags_2_prob_padded.shape[0]):
                full_emission = pair_tags_2_prob_padded[idx]
                full_mask = pair_tags_mask_tensor[idx].byte()
                full_tags = torch.tensor(pred_pair_rep_args_tag_2[idx]).cuda()
                pred_pair_rep_args_probs = []
                for pred_span in pred_pair_rep_args_2_list[idx]:
                    start, end = pred_span
                    emission = full_emission[start: end+1].unsqueeze(1)  # [1, end-start+1]
                    mask = full_mask[start: end+1].unsqueeze(1)    # [1, end-start+1]
                    tags = full_tags[start:end+1].unsqueeze(1)  # [1, end-start+1]
                    numerator = self.pair_2_crf._compute_score(emission, tags, mask)
                    denominatr = self.pair_2_crf._compute_normalizer(emission, mask)
                    llh = numerator - denominatr
                    prob = llh.exp()
                    pred_pair_rep_args_probs.append(prob.tolist()[0])
                pred_pair_rep_args_prob_2_list.append(pred_pair_rep_args_probs)

            i = 0
            for pred_arguments in pred_rep_args_list:
                pred_args_pair_2_dict = {}
                for args in pred_arguments:
                    pred_args_pair_2_dict[args] = (pred_pair_rep_args_2_list[i], \
                                                    pred_pair_rep_args_prob_2_list[i])
                    i+=1
                pred_args_pair_2_dict_list.append(pred_args_pair_2_dict)

        return pred_rev_args_list, pred_rep_args_list, pred_args_pair_dict_list, pred_args_pair_2_dict_list

    
    def padding_and_mask(self, ids_list):
        max_len = max([len(x) for x in ids_list])
        mask_list = []
        ids_padding_list = []
        for ids in ids_list:
            mask = [1.] * len(ids) + [0.] * (max_len - len(ids))
            ids = ids + [0] * (max_len - len(ids))
            mask_list.append(mask)
            ids_padding_list.append(ids)
        return ids_padding_list, mask_list
