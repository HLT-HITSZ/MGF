import re
from transformers import BertModel, BertTokenizer
import numpy as np
import json

processed_root_dir = 'data/processed_rr_sub/'
special_tokens = ['[ENDL]', '[TAB]', '[LINE]',
                    '[EQU]', '[URL]', '[NUM]',
                    '[SPE]']
tokenizer = BertTokenizer.from_pretrained('./data/bert-base-uncased',
                        additional_special_tokens=special_tokens)
if __name__ == "__main__":
    max_len = 0
    sent_len_list = []
    cnt = 0
    for data_file in ['dev.txt', 'test.txt', 'train.txt']:
    # for data_file in ['dev.txt']:
        with open(processed_root_dir + data_file, 'r') as fp:
            sample_list = fp.read().split('\n\n\n')
            print(data_file + ':' + str(len(sample_list)))
        
        sent_num_list = []
        with open(processed_root_dir + data_file + '.bioes', 'w') as fp:
            graph_list = json.load(open(processed_root_dir + data_file[:-4] + '.graph.json', 'r'))
            new_graph_list = []
            for sample, graph in zip(sample_list, graph_list):
                if sample == '':
                    continue

                review_text, reply_text = sample.split('\n\n')
                sent_num_list.append(len(review_text.split('\n')))
                sent_num_list.append(len(reply_text.split('\n')))

                line_list = review_text.split('\n')
                rev_have_args_flag = False
                rev_output_str = ''
                for idx in range(len(line_list)):
                    if idx == len(line_list) - 1:
                        next_bio = None
                    else:
                        next_bio = line_list[idx+1].split('\t')[1]
                    
                    tmp = line_list[idx].strip().split('\t')
                    sent = tmp[0]
                    bio_tag = tmp[1]
                    pair_tag = tmp[2]
                    text_type = tmp[3]
                    sub_id = tmp[4]

                    bioes_tag = None
                    if bio_tag == 'B-Review':
                        rev_have_args_flag = True
                        if next_bio == 'B-Review':
                            bioes_tag = 'S-Review'
                            pair_tag = 'S' + pair_tag[1:]
                        elif next_bio == 'O':
                            bioes_tag = 'S-Review'
                            pair_tag = 'S' + pair_tag[1:]
                        elif next_bio == 'I-Review':
                            bioes_tag = 'B-Review'
                            pair_tag = 'B' + pair_tag[1:]
                        elif next_bio == None:
                            bioes_tag = 'S-Review'
                            pair_tag = 'S' + pair_tag[1:]
                    elif bio_tag == 'I-Review':
                        rev_have_args_flag = True
                        if next_bio == 'B-Review':
                            bioes_tag = 'E-Review'
                            pair_tag = 'E' + pair_tag[1:]
                        elif next_bio == 'O':
                            bioes_tag = 'E-Review'
                            pair_tag = 'E' + pair_tag[1:]
                        elif next_bio == 'I-Review':
                            bioes_tag = 'I-Review'
                            pair_tag = 'I' + pair_tag[1:]
                        elif next_bio == None:
                            bioes_tag = 'E-Review'
                            pair_tag = 'E' + pair_tag[1:]
                    else:
                        bioes_tag = 'O'
                        pair_tag = 'O'

                    rev_output_str += '{}\t{}\t{}\t{}\t{}\n'.format(
                        sent, bioes_tag, pair_tag, text_type, sub_id)

                rep_have_args_flag = False
                rep_output_str = ''
                line_list = reply_text.split('\n')
                for idx in range(len(line_list)):
                    if idx == len(line_list) - 1:
                        next_bio = None
                    else:
                        next_bio = line_list[idx+1].split('\t')[1]
                    
                    tmp = line_list[idx].strip().split('\t')
                    sent = tmp[0]
                    bio_tag = tmp[1]
                    pair_tag = tmp[2]
                    text_type = tmp[3]
                    sub_id = tmp[4]

                    bioes_tag = None
                    if bio_tag == 'B-Reply':
                        rep_have_args_flag = True
                        if next_bio == 'B-Reply':
                            bioes_tag = 'S-Reply'
                            pair_tag = 'S' + pair_tag[1:]
                        elif next_bio == 'O':
                            bioes_tag = 'S-Reply'
                            pair_tag = 'S' + pair_tag[1:]
                        elif next_bio == 'I-Reply':
                            bioes_tag = 'B-Reply'
                            pair_tag = 'B' + pair_tag[1:]
                        elif next_bio == None:
                            bioes_tag = 'S-Reply'
                            pair_tag = 'S' + pair_tag[1:]
                    elif bio_tag == 'I-Reply':
                        rep_have_args_flag = True
                        if next_bio == 'B-Reply':
                            bioes_tag = 'E-Reply'
                            pair_tag = 'E' + pair_tag[1:]
                        elif next_bio == 'O':
                            bioes_tag = 'E-Reply'
                            pair_tag = 'E' + pair_tag[1:]
                        elif next_bio == 'I-Reply':
                            bioes_tag = 'I-Reply'
                            pair_tag = 'I' + pair_tag[1:]
                        elif next_bio == None:
                            bioes_tag = 'E-Reply'
                            pair_tag = 'E' + pair_tag[1:]
                    else:
                        bioes_tag = 'O'
                        pair_tag = 'O'

                    rep_output_str += '{}\t{}\t{}\t{}\t{}\n'.format(
                        sent, bioes_tag, pair_tag, text_type, sub_id)

                if rev_have_args_flag == False or rep_have_args_flag == False:
                    cnt+=1
                if data_file == 'train.txt':
                    if rev_have_args_flag and rep_have_args_flag:
                        fp.write(rev_output_str)
                        fp.write('\n')
                        fp.write(rep_output_str)
                        fp.write('\n\n')
                        new_graph_list.append(graph)
                else:
                    fp.write(rev_output_str)
                    fp.write('\n')
                    fp.write(rep_output_str)
                    fp.write('\n\n')
                    new_graph_list.append(graph)
            json.dump(new_graph_list, open(processed_root_dir + data_file[:-4] + '.filtered.graph.json', 'w'))
            pass
            print(cnt)
pass