#!/usr/bin/env python
# -*- coding: utf-8 -*-
#/home/celia/Traditional_Chinese_Medicine_MyProject/Utils/utils.py
import random
from pathlib import Path
import os
import time
import logging

def master_print(save_var, name_print):
    log_file = open(f"print_{name_print}.txt","a+")
    print('\n',save_var, file=log_file)
    log_file.close()
    print(error)

def scores_renew(scores, model_names, score, mode_name):
    scores_renew = scores
    model_names_renew = model_names
    flag_renew = False
    for i, s in enumerate(scores):
        if score > s:

            scores_renew.insert(i, score)
            scores_renew.pop(-1)
            model_names_renew.insert(i, mode_name)
            model_names_renew.pop(-1)
            os.system(f"rm -f {model_names[-1]}*")  #delete former
            flag_renew = True
            break
        
    scores = scores_renew
    model_names = model_names_renew
    return scores, model_names, flag_renew

def convert_single_example(tokenizer, text_a):
    """pramas:
    tokenizer--> tokenization.FullTokenizer(vocab_file=vocab_file, do_lower_case=False)
    text_a --> a batch text
    return --> tokens, 
            input_ids, 
            input_mask
    对应的就是创建bert模型时候的segment_ids, input_ids, input_mask参数
    """
    def _tokenize(token_dict,text):
        R = []
        for c in text:
            if c in token_dict:
                R.append(c)
            elif c==' ':
            # in original bert Space unrecognized, so replace it
                R.append('[unused1]') # space类用未经训练的[unused1]表示
            else:
                R.append('[UNK]') # 剩余的字符是[UNK]
        return R
    
    tokens_a = _tokenize( tokenizer.vocab, text_a)

    # 如果只有一个句子，只用在前后加上[CLS], [SEP] 所以句子长度要小于 max_seq_length - 2
    tokens = []
    tokens.append("[CLS]")
    for token in tokens_a:
        tokens.append(token)
    tokens.append("[SEP]")
    

    input_ids = tokenizer.convert_tokens_to_ids(tokens)# 将中文转换成ids
    # master_print(input_ids, "input_ids")
    input_mask = [1] * len(input_ids) # 创建mask
    
    return tokens, input_ids, input_mask 
    

def process(text, args, tokenizer):
    
    tokens, t1, t2 = convert_single_example(tokenizer, text)
    t1 = t1 + [0] * (args.max_x_length + 2 - len(t1)) #add start&end
    t2 = t2 + [0] * (args.max_x_length + 2 - len(t2))
    
    return tokens, t1, t2


class Batch:
    #batch类，里面包含了encoder输入，decoder输入以及他们的长度
    def __init__(self):
        self.input_x_word = []
        self.input_mask = []
        self.inputs_y = []
        self.sequence_lengths = []


def createBatch(samples, args, tokenizer):

    batch = Batch()
    for sample in samples:
        #将source PAD值本batch的最大长度
        tokens, t1, t2 = process(sample[0], args, tokenizer)
        batch.input_x_word.append(t1)
        batch.input_mask.append(t2)
        batch.sequence_lengths.append(sample[1])
        # batch.sequence_lengths.append(sample[1]+2) #why+2? ==>CLS SEP denote the start&end of one sentence
        
        y = list(sample[2])
        #predict
        batch.inputs_y.append( [ args.category2id["[CLS]"] ] + y + [ args.category2id["[SEP]"] ] + 
                              [ args.category2id["O"] ] * (args.max_x_length - len(y)) )
        
        # batch.inputs_y.append(  y + [args.category2id["O"] ] * (args.max_x_length - len(y)) )
        
    return batch


def getBatches(data, args, tokenizer):
    #每个epoch之前都要进行样本的shuffle
    random.shuffle(data)
    batches = []
    data_len = len(data)
    def genNextSamples():
        for i in range(0, data_len, args.batch_size):
            yield data[i:min(i + args.batch_size, data_len)]

    for samples in genNextSamples():
        batch = createBatch(samples, args, tokenizer)
        batches.append(batch)

    return batches
    
def create_logger(cfg, cfg_name, phase='train'):
    root_output_dir = Path(cfg.MODEL_SAVE_ROOT)
    # set up logger
    if not root_output_dir.exists():
        print('=> creating {}'.format(root_output_dir))
        root_output_dir.mkdir()

    time_str = time.strftime('%Y-%m-%d-%H-%M')
    log_file = '{}_{}_{}.log'.format(cfg_name, time_str, phase)
    final_log_file = final_output_dir / log_file
    head = '%(asctime)-15s %(message)s'
    logging.basicConfig(filename=str(final_log_file),
                        format=head)

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    console = logging.StreamHandler()
    logging.getLogger('').addHandler(console)
    
    return logger, str(final_output_dir)


