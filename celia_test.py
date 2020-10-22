#!/usr/bin/env python
# -*- coding: utf-8 -*-
import json
import numpy as np
from tqdm import tqdm
import os
import pandas as pd
import tensorflow as tf
from tensorflow.contrib.crf import viterbi_decode

from Bert.bert import tokenization
from Utils.model import entity_model
from Cfg_files import cfg

from datetime import datetime

from Utils.utils import process
from Utils.data_utils import load_testData, ssbsTest, data_preprocess

import math

def NewtonCoolingLaw(alpha,t):
    value = math.exp(-alpha*t)
    return value

def loadModel(args, model_pool, fold_idx):
    fold_idx = str(fold_idx)
    tf.reset_default_graph()
    session = tf.Session()
    model = entity_model(args)
    saver = tf.train.Saver(tf.global_variables(), max_to_keep=args.num_checkpoints)
    path = os.path.join(model_pool, fold_idx) #model_saved/0
    print(f'load model from: {path}')
    
    """About tf.train.get_checkpoint_state
        function: Returns CheckpointState proto from the "checkpoint" file.
        param_1: checkpoint_dir, 
        param_2: latest_filename=None #Optional name of the checkpoint file. Default to 'checkpoint'.
        return: A CheckpointState if the state was available, None otherwise. """
    ckpt = tf.train.get_checkpoint_state(path)
    if ckpt and ckpt.model_checkpoint_path:
        saver.restore(session, ckpt.model_checkpoint_path)
    # print(f'==> ckpt.model_checkpoint_path,{ckpt.model_checkpoint_path}')    
    f1 = "0."+ckpt.model_checkpoint_path.split('/')[-1][20:24]
    print('##########',f'f1:{f1}')

    return model, session, float(f1)


def logits_trans(model, session, t1, t2, d):
    feed_dict = {
        model.input_x_word: [t1],
        model.input_mask: [t2],
        model.input_x_len: [ len(d['tx'])+2 ],
        model.keep_prob: 1,
        model.is_training: False,
    }
    lengths, logits, trans = session.run(
        fetches=[model.lengths, model.logits, model.trans],
        feed_dict=feed_dict
        )
    return lengths, logits, trans


def decode(logits, lengths, matrix, args):
    """ :param logits: [batch_size, num_steps, num_tags] float32, logits
        :param lengths: [batch_size] int32, real length of each sequence
        :param matrix: transaction matrix for inference
        :return: paths list"""
    # inference final labels use viterbi Algorithm
    paths = []
    small = -1000.0
    start = np.asarray([[small] * args.relation_num + [0]])
    print('length:', lengths)

    for score, length in zip(logits, lengths):
        score = score[:length]
        pad = small * np.ones([length, 1])
        logits = np.concatenate([score, pad], axis=1)
        logits = np.concatenate([start, logits], axis=0)

        """ About =viterbi_decode=
        Decode the highest scoring sequence of tags outside of TensorFlow.
        This should only be used at test time.
        Args:
            score: A [seq_len, num_tags] matrix of unary potentials.
            transition_params: A [num_tags, num_tags] matrix of binary potentials.

        Returns:
            viterbi: A [seq_len] list of integers containing the highest scoring tag indices.
            viterbi_score: A float containing the score for the Viterbi sequence. """
        path, _ = viterbi_decode(logits, matrix)
        
        paths.append(path[1:])

    return paths

def master_print(save_var, name_print):
    log_file = open(f"print_{name_print}.txt","a+")
    print('\n',save_var, file=log_file)
    log_file.close()

if __name__=='__main__':
    args = cfg.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    print(f'assign gpu={os.environ["CUDA_VISIBLE_DEVICES"]}')

    model_saved_file = args.model_to_test
    print("test model==>", model_saved_file) #"/dataset/nlp_ckpt_celia/20201010/Experiments/10-10_normal/model_saved"

    experiment_name =model_saved_file.split('/')[4] #10-10-normal
    print(f"\nexperiment_name: {experiment_name}")
    experiment_root = model_saved_file.replace("/model_saved",'') #"/dataset/nlp_ckpt_celia/20201010/Experiments/10-10_normal
     
    log_name = experiment_name + '.txt'
    log_dir = experiment_root + '/' + log_name
    print('\nlog_dir=',log_dir)
    
    args.mode = "test"
    """ experiment record """
    cfg.record_args(log_dir, args)

    """bert"""
    args.vocab_file = args.bert_root + 'vocab.txt'
    args.bert_config_file = args.bert_root + 'bert_config.json'
    args.bert_file = args.bert_root + 'bert_model.ckpt'

    category2id, id2category = data_preprocess()
    args.relation_num = len(category2id)
    args.category2id = category2id

    """load data"""
    test = load_testData()  # read Txt
    test_data = ssbsTest(test, args.max_x_length)
    master_print(test_data,"test_data")

    """see see"""
    # print_txt = open(args.print_sth,"a+")
    # print('=====test_data===== \n',test_data,'\n',
    #     file=print_txt)
    # print_txt.close()

    """ FullTokenizer include: BasicTokenizer & WordpieceTokenizer
        return: split_tokens """
    tokenizer = tokenization.FullTokenizer(vocab_file=args.vocab_file, 
                                            do_lower_case=False)
    
    model_pool = model_saved_file #model_saved
    
    model0, session0, f1_0 = loadModel(args, model_pool, fold_idx=0)
    model1, session1, f1_1 = loadModel(args, model_pool, fold_idx=1)
    model2, session2, f1_2 = loadModel(args, model_pool, fold_idx=2)
    model3, session3, f1_3 = loadModel(args, model_pool, fold_idx=3)
    model4, session4, f1_4 = loadModel(args, model_pool, fold_idx=4)
    model5, session5, f1_5 = loadModel(args, model_pool, fold_idx=5)
    model6, session6, f1_6 = loadModel(args, model_pool, fold_idx=6)
    model7, session7, f1_7 = loadModel(args, model_pool, fold_idx=7)
    model8, session8, f1_8 = loadModel(args, model_pool, fold_idx=8)
    model9, session9, f1_9 = loadModel(args, model_pool, fold_idx=9)
    model10, session10, f1_10 = loadModel(args, model_pool, fold_idx=10)
    f1_list = [ f1_0, f1_1, f1_2, f1_3, f1_4, f1_5, f1_6, f1_7,f1_8,f1_9,f1_10 ]
    # f1_list = [ f1_0, f1_1, f1_2, f1_3, f1_4, f1_5, f1_6 ]
    # f1_list = [ f1_0, f1_1, f1_2, f1_3, f1_4, f1_5]
    # f1_list = [ f1_0, f1_1, f1_2, f1_3, f1_4]
    # f1_list = [ f1_0, f1_1]

    f1 = sorted(f1_list,reverse = True)

    wf1 = [NewtonCoolingLaw(args.alpha,t) for t in range(len(f1))]
    print(f"wf1: {wf1}")

    for iid in tqdm(set(test_data['id'])):
        sample = test_data[test_data['id'] == str(iid)]
        master_print(sample,"sample")
        pred_list = []
        num = 1
        for d in sample.iterrows():
            d = d[1]
            master_print(d,"d")

            #d['tx'] is text
            master_print(d['tx'],"d['tx']")
            tokens, t1, t2 = process(d['tx'], args, tokenizer)

            master_print(tokens,"tokens")
            master_print(t1,"t1")
            master_print(t2,"t2")

            
            # for mode in range(args.k_folds):
            for mode in range(11):
                if mode == 0:
                    # get predict from model
                    lengths, logits, trans = logits_trans(model0, session0, t1, t2, d)
                    w = NewtonCoolingLaw (args.alpha, f1.index(f1_0))
                    master_print(lengths,"lengths")
                    master_print(logits,"logits")
                    master_print(trans,"trans")
                elif mode == 1:
                    lengths, logits, trans = logits_trans(model1, session1, t1, t2, d)
                    w = NewtonCoolingLaw (args.alpha, f1.index(f1_1))
                elif mode == 2:
                    lengths, logits, trans = logits_trans(model2, session2, t1, t2, d)
                    w = NewtonCoolingLaw (args.alpha, f1.index(f1_2))
                elif mode == 3:
                    lengths, logits, trans = logits_trans(model3, session3, t1, t2, d)
                    w = NewtonCoolingLaw (args.alpha, f1.index(f1_3))
                elif mode == 4:
                    lengths, logits, trans = logits_trans(model4, session4, t1, t2, d)
                    w = NewtonCoolingLaw (args.alpha, f1.index(f1_4))
                elif mode == 5:
                    lengths, logits, trans = logits_trans(model5, session5, t1, t2, d)
                    w = NewtonCoolingLaw (args.alpha, f1.index(f1_5))
                elif mode == 6:
                    lengths, logits, trans = logits_trans(model6, session6, t1, t2, d)
                    w = NewtonCoolingLaw (args.alpha, f1.index(f1_6))
                elif mode == 7:
                    lengths, logits, trans = logits_trans(model7, session7, t1, t2, d)
                    w = NewtonCoolingLaw (args.alpha, f1.index(f1_7))
                elif mode == 8:
                    lengths, logits, trans = logits_trans(model8, session8, t1, t2, d)
                    w = NewtonCoolingLaw (args.alpha, f1.index(f1_8))
                elif mode == 9:
                    lengths, logits, trans = logits_trans(model9, session9, t1, t2, d)
                    w = NewtonCoolingLaw (args.alpha, f1.index(f1_9))
                elif mode == 10:
                    lengths, logits, trans = logits_trans(model10, session10, t1, t2, d)
                    w = NewtonCoolingLaw (args.alpha, f1.index(f1_10))
                
                
                if mode == 0:
                    logits_ = logits
                    trans_ = trans
                else:
                    logits_ += logits * w
                    trans_ += trans * w
                    
            logits_ = logits_ / np.sum(wf1)
            trans_ = trans_ / np.sum(wf1)
            
            pred = decode(logits_, lengths, trans_, args)[0]
            master_print(pred,"predfromdecode")
            pred = [ id2category[w] for w in pred ]
            master_print(pred,"pred_2nd")


            for offset,p in enumerate( pred ):
                if p[0] == 'B':
                    endPos = offset+1
                    for i in range(1,10):
                        if pred[offset+i][0]=='I':
                            endPos = offset+i
                        else:
                            break
                    startPos_ = d['lineStartPosition'] + offset-1
                    endPos_ = d['lineStartPosition'] + endPos
                    pred_list.extend( [( 'T{0}'.format(num), p[2:]+' '+str(startPos_)+' '+str(endPos_), ''.join(tokens[offset:endPos+1]) )] )
                    num += 1
                if p[0] == 'S':
                    startPos_ = d['lineStartPosition'] + offset-1
                    endPos_ = d['lineStartPosition'] + offset
                    pred_list.extend( [( 'T{0}'.format(num), p[2:]+' '+str(startPos_)+' '+str(endPos_), ''.join(tokens[offset:offset+1]) )] )
                    num += 1
                                        
        pred_list = pd.DataFrame(pred_list)
        master_print(pred_list,"df_pred_list")

        # """see see"""
        # print_txt = open(args.print_sth,"a+")
        # print('=====pred_list===== \n',pred_list,'\n',
        #     file=print_txt)
        # print_txt.close()
        
        print(f"\nexperiment_root:{experiment_root}")
        """ save """
        predict_result = os.path.join(experiment_root, args.pred_results)
        print(f"\npredict_result: {predict_result}")
        pre_path = predict_result + "_" + datetime.now().strftime('%m%d')
        if not os.path.exists(pre_path):
            os.mkdir(pre_path)
        print(f"\npre_path: {pre_path}")
        
        pred_list.to_csv(pre_path + '/{0}.ann'.format(iid), encoding='utf8', header=False, sep='\t', index=False)

        log_txt = open(log_dir,"a+")
        print(f'finish test, save results in : {pre_path}', file=log_txt)
        log_txt.close()

    print(datetime.now(), 'Test Finished!')
    