import re
from transformers import BertModel, BertTokenizer
import numpy as np
from tqdm import tqdm
from collections import Counter
import json
import os

max_co_occur_words_num = 46

root_dir = 'data/rr-submission/'
processed_root_dir = 'data/processed_rr_sub/'
if not os.path.exists(processed_root_dir):
    os.mkdir(processed_root_dir)

special_tokens = ['[ENDL]', '[TAB]', '[LINE]',
                    '[EQU]', '[URL]', '[NUM]',
                    '[SPE]']
tokenizer = BertTokenizer.from_pretrained('./data/bert-base-uncased',
                        additional_special_tokens=special_tokens)

def preprocess(sent):
    sent = sent.replace('[line_break_token]', ' [ENDL] ')
    sent = sent.replace('[tab_token]', ' [TAB] ')
    sent = re.sub('[+-]{3,}', ' [LINE] ', sent).strip()
    sent = re.sub('={2,}', ' [EQU] ', sent).strip()
    sent = re.sub('_{2,}', ' [LINE] ', sent).strip()
    sent = re.sub('<[^>]+>[^>]+>', ' [URL] ', sent).strip()
    sent = re.sub('[0-9]+\.[0-9]+', ' [NUM] ', sent).strip()
    sent = re.sub('(Äî‚)+', ' [SPE] ', sent).strip()

    sent = re.sub(' +', ' ', sent).strip()
    
    token_list = tokenizer.tokenize(sent)
    token_list = token_list[:50]
    token_list = ['[CLS]'] + token_list + ['[SEP]']
    token_sent = ' '.join(token_list)

    return token_sent

max_len = 0
sent_len_list = []

char_counter = Counter()

for data_file in ['dev.txt', 'test.txt', 'train.txt']:
# for data_file in ['dev.txt']:
    with open(root_dir + data_file, 'r') as fp:
        raw_sample_list = fp.read().split('\n\n')
        print(data_file + ':' + str(len(raw_sample_list)))

    
    graph_fw = open(processed_root_dir + data_file[:-3] + 'graph.json', 'w')

    with open(processed_root_dir + data_file, 'w') as fp:
        sample_dict_list = []
        for raw_sample in raw_sample_list:
            if raw_sample == '':
                continue
            line_list = raw_sample.split('\n')
            sample_dict = {'review': {'sent_ids': [], 'sents': [], 'bio_tag': [], 'pair_tag': [], 'sub_id': None}, 
                                'reply': {'sent_ids': [], 'sents': [], 'bio_tag': [], 'pair_tag': [], 'sub_id': None},
                                'graph': set()}

            rev_idx = -1
            rep_idx = -1
            total_idx = -1
            for idx, line in enumerate(line_list):
                line = re.sub('\.\t', ' .\t', line).strip()
                line = re.sub('!\t', ' !\t', line).strip()
                line = re.sub('\?\t', ' ?\t', line).strip()
                tmp = line.strip().split('\t')

                char_counter += Counter(tmp[0])

                text_type = tmp[3].lower()
                if text_type == 'review':
                    if rev_idx >= 99:
                        # print("hhh")
                        continue
                    else:
                        rev_idx += 1
                        total_idx += 1
                else:
                    if rep_idx >= 99:
                        # print("hhh")
                        continue
                    else:
                        rep_idx += 1
                        total_idx += 1
  
                sample_dict[text_type]['sent_ids'].append(total_idx)
                sample_dict[text_type]['sents'].append(tmp[0])
                sample_dict[text_type]['bio_tag'].append(tmp[1])
                sample_dict[text_type]['pair_tag'].append(tmp[2])
                sample_dict[text_type]['sub_id'] = tmp[4]
            sample_dict_list.append(sample_dict)

            # fp.write('\n\n')
        
        max_dist = 1
        for sample_dict in tqdm(sample_dict_list):
            for sent_id_x in sample_dict['review']['sent_ids']:
                for sent_id_y in sample_dict['review']['sent_ids']:
                    if sent_id_x == sent_id_y:
                        sample_dict['graph'].add((sent_id_x, sent_id_y, 2.0))
                        break
                    elif 0 < sent_id_x-sent_id_y <= max_dist:
                        weight = 2.0 - (sent_id_x-sent_id_y)/max_dist
                        sample_dict['graph'].add((sent_id_x, sent_id_y, weight))
                        sample_dict['graph'].add((sent_id_y, sent_id_x, weight))
                    elif 0 > sent_id_x-sent_id_y:
                        print("??")
            
            for sent_id_x in sample_dict['reply']['sent_ids']:
                for sent_id_y in sample_dict['reply']['sent_ids']:
                    if sent_id_x == sent_id_y:
                        sample_dict['graph'].add((sent_id_x, sent_id_y, 2.0))
                        break
                    elif 0 < sent_id_x-sent_id_y <= max_dist:
                        weight = 2.0 - (sent_id_x-sent_id_y)/max_dist
                        sample_dict['graph'].add((sent_id_x, sent_id_y, weight))
                        sample_dict['graph'].add((sent_id_y, sent_id_x, weight))
                    elif 0 > sent_id_x-sent_id_y:
                        print("??")
            
            def clean_sent(sent):
                sent = sent.lower()
                sent = sent.replace('[line_break_token]', ' ')
                sent = sent.replace('[tab_token]', ' ')
                sent = re.sub(', ', ' , ', sent).strip()
                sent = re.sub(': ', ' : ', sent).strip()
                sent = re.sub('; ', ' ; ', sent).strip()
                sent = re.sub('\*', ' ', sent).strip()
                sent = re.sub('" ', ' " ', sent).strip()
                sent = re.sub(' "', ' " ', sent).strip()
                sent = re.sub(" '", " ' ", sent).strip()
                sent = re.sub("' ", " ' ", sent).strip()
                sent = re.sub("\) ", " ) ", sent).strip()
                sent = re.sub(" \(", " ( ", sent).strip()

                sent = re.sub(' +', ' ', sent).strip()
                return sent
            
            def remove_stopwords(word_list):
                from nltk.corpus import stopwords
                stopwords = set(stopwords.words('english'))
                for w in ['!',',','.','?', '(', ')', '"', "'", ';', ':']:
                    stopwords.add(w)
                filtered_words = [word for word in word_list if word not in stopwords]
                return filtered_words
            
            for x, sent_rev in enumerate(sample_dict['review']['sents']):
                sent_rev = clean_sent(sent_rev)
                sent_rev_tokens = sent_rev.split(' ')
                sent_rev_tokens = remove_stopwords(sent_rev_tokens)
                words_rev_dict = dict(Counter(sent_rev_tokens))
                # if sample_dict['review']['pair_tag'][x] != 'O':
                #     print('hh')
                #     pass
                for y, sent_rep in enumerate(sample_dict['reply']['sents']):
                    sent_rep = clean_sent(sent_rep)
                    sent_rep_tokens = sent_rep.split(' ')
                    sent_rep_tokens = remove_stopwords(sent_rep_tokens)
                    words_rep_dict = dict(Counter(sent_rep_tokens))
                    co_occur_words = words_rev_dict.keys() & words_rep_dict.keys()
                    co_occur_words_dict = {word: words_rev_dict[word] + words_rep_dict[word] for word in co_occur_words}
                    if len(co_occur_words_dict) >= 2:
                        # if len(co_occur_words_dict) >= max_co_occur_words_num:
                        #     weight = 2.0
                        # else:
                        weight = len(co_occur_words_dict) / max_co_occur_words_num + 1
                        sample_dict['graph'].add((sample_dict['review']['sent_ids'][x], \
                                            sample_dict['reply']['sent_ids'][y], weight))
                        sample_dict['graph'].add((sample_dict['reply']['sent_ids'][y], \
                                            sample_dict['review']['sent_ids'][x], weight))

        graph_list = []
        for sample_dict in sample_dict_list:
            review_dict, reply_dict, graph = sample_dict['review'], sample_dict['reply'], sample_dict['graph']
            sub_id = review_dict['sub_id']
            for token_sent, bio_tag, pair_tag in zip(review_dict['sents'], \
                                                review_dict['bio_tag'], \
                                                review_dict['pair_tag']):
                token_sent = preprocess(token_sent)
                fp.write('{}\t{}\t{}\t{}\t{}\n'.format(
                    token_sent, bio_tag, pair_tag, 'Review', sub_id))
            
            fp.write('\n')
            
            for token_sent, bio_tag, pair_tag in zip(reply_dict['sents'], \
                                                reply_dict['bio_tag'], \
                                                reply_dict['pair_tag']):
                token_sent = preprocess(token_sent)
                fp.write('{}\t{}\t{}\t{}\t{}\n'.format(
                    token_sent, bio_tag, pair_tag, 'Reply', sub_id))         

            fp.write('\n\n')
            graph_list.append(list(graph))
        json.dump(graph_list, graph_fw)
        graph_fw.close()
pass
