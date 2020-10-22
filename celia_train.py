#/home/celia/Traditional_Chinese_Medicine_MyProject/train.py
import os
import json
import numpy as np
from tqdm import tqdm
from datetime import datetime
import argparse

import tensorflow as tf

from Cfg_files import cfg
from Bert.bert import tokenization
from Utils.model import entity_model
from Utils.utils import getBatches
from Utils import data_utils
from Utils.optimization import create_optimizer
from Utils.eval_metrics import getScore
from Utils.utils import scores_renew

print('current path==> ', os.getcwd())

def Train(args):
    """ load data from original dataset"""
    """all *.txt & *.ann pairs in args.baseTrainPath, totally 1000*2 items"""
    total_data, dfLabel = data_utils.load_data(args.baseTrainPath)
    
    """total_data : dict {"id":"content",...} """
    """dfLabel : pd.DataFrame {num, id, entity, category, pe1, pe2}
        num--sequence number;
        id--corresponding txtfilename;
        entity-- keywords eg:糖衣片,味甘
        category-- eg: DRUG_DOSAGE,DRUG_TASTE
        pe1,pe2 -- start and end position
    """
    
    if not os.path.exists(args.devided_data):
        print("==recreate a devided_data")
        random_order = list(range(len(total_data)))
        np.random.shuffle(random_order)
        json.dump(
            random_order,
            open(args.devided_data, 'w'),
            indent=4 #make output elegantly
        )
    else:
        print(f'==load data from: {args.devided_data}')
        random_order = json.load(open(args.devided_data))
        # print(f'total: {len(random_order)}') #1000

    """ FullTokenizer include: BasicTokenizer & WordpieceTokenizer
        return: split_tokens """
    tokenizer = tokenization.FullTokenizer(vocab_file=args.vocab_file, do_lower_case=False)

    category2id, id2category = data_utils.data_preprocess()
    

    args.relation_num = len(category2id)
    args.category2id = category2id

    k_folds = args.k_folds
    for mode in range(k_folds):
        # record
        log_file = open(log_dir,"a+")
        print('==mode==>',mode, file=log_file)
        log_file.close()

        train_data = {}
        dev_data = {}

        # devided data into train&dev set
        for i in random_order:
            if i % k_folds != mode:
                train_data[str(i)] = total_data[str(i)] # train_data[str(i)] is a sentence
            else:
                dev_data[str(i)] = total_data[str(i)]
       
        # print_txt = open(args.print_sth,"a+")
        # print('\n===train_data===\n', train_data, '\n', file=print_txt)
        # print_txt.close()

        print("train_data : %d" % len (train_data)) 
        print("dev_data : %d" % len (dev_data)) 

        train_data = data_utils.ssbs(train_data, dfLabel, args.max_x_length)
        dev_data = data_utils.ssbs(dev_data, dfLabel, args.max_x_length)
    
        train_data = data_utils.trans2data(train_data, category2id, args.max_x_length)

        # print_txt = open(args.print_sth,"a+")
        # print('\n===train_data===\n', train_data, '\n', file=print_txt)
        # print_txt.close()


        batchesTrain = getBatches(train_data, args, tokenizer)


        
        for nextBatch in batchesTrain:
            """ show address  Q:how to show directly? """
            print_txt = open(args.print_sth,"a+")
            print('==nextBatch===\n', nextBatch, file=print_txt)
            print_txt.close()
            break
    
        dev_data = data_utils.trans2data(dev_data, category2id, args.max_x_length)
        

        # """see see"""
        # print_txt = open(args.print_sth,"a+")
        # print('==dev_data===\n', dev_data, file=print_txt)
        # print_txt.close() 
        
        print ('\n...start training...')
        graph = tf.Graph()
        with graph.as_default():
            """configure Session settings"""
            session_conf = tf.ConfigProto(allow_soft_placement=True, # if true, allow tf assign device automatively.
                                        log_device_placement=True, # if true, print log about device assignment
                                        device_count = {'GPU': 1}) #= dict, assign CPU_id and GPU_id
            # assign memory dynamicly. if true, at beginning,assign few, then increase on-demand
            session_conf.gpu_options.allow_growth = True 
            session = tf.Session(config=session_conf)
            
            with session.as_default():
                """model"""
                # TODO -->model.loss
                model = entity_model(args)
                ####

                # = Q:how can I print model specifically
                # log_txt = open(args.log_file,"a+")
                # print(f"model:\n {model}",file=log_txt)  
                # log_txt.close()

                """training setting"""
                global_step = tf.Variable(0, name='step', trainable=False)

                learning_rate = tf.train.exponential_decay(args.learning_rate, 
                                                            global_step, 
                                                            args.decay_step,
                                                            args.decay_rate, 
                                                            staircase=True)
                
                normal_optimizer = tf.train.AdamOptimizer(learning_rate)  # 下接结构的学习率

                # tf.train.write_graph(graph.as_graph_def(),'.','graph.pb',False) #= save
                
                all_variables = graph.get_collection('trainable_variables')
                word2vec_var_list = [x for x in all_variables if 'bert' in x.name]  # BERT的参数
                normal_var_list = [x for x in all_variables if 'bert' not in x.name]  # 下接结构的参数

                # print('bert train variable num: {}'.format(len(word2vec_var_list))) # 199
                # print('normal train variable num: {}'.format(len(normal_var_list))) # 29

                normal_op = normal_optimizer.minimize(model.loss, 
                                                    global_step=global_step, 
                                                    var_list=normal_var_list)
                num_batch = int( len(train_data) / args.batch_size * args.train_epoch)
                print(f"num_batch: {num_batch}")
                
                embed_step = tf.Variable(0, name='step', trainable=False)
                if word2vec_var_list: 
                    print('finetuning~~~~')
                    word2vec_op, embed_learning_rate, embed_step = create_optimizer(model.loss, 
                                                                                    args.embed_learning_rate, 
                                                                                    num_train_steps=num_batch,
                                                                                    num_warmup_steps=int(num_batch * 0.05), 
                                                                                    use_tpu=False, 
                                                                                    variable_list=word2vec_var_list)
        
                    train_op = tf.group(normal_op, word2vec_op) 
                else:
                    train_op = normal_op
        
                saver = tf.train.Saver(tf.global_variables(), max_to_keep=args.num_checkpoints)
                
                #resume
                if args.continue_training:
                    print('recover from: {}'.format(args.checkpointPath))
                    saver.restore(session, args.checkpointPath)
                else:
                    session.run(tf.global_variables_initializer())
                
                save_ckpt_num = args.save_ckpt_num
                scores = [0] * save_ckpt_num
                model_names = ["cache"] * save_ckpt_num
                
                current_step = 0
                for e in range(args.train_epoch):
                    print("----- Epoch {}/{} -----".format(e + 1, args.train_epoch))

                    log_txt = open(log_dir,"a+")
                    print(str(datetime.now())[0:16],
                        "----- Epoch {}/{} -----".format(e + 1, args.train_epoch),
                        file=log_txt)
                    log_txt.close()
                    
                    loss_ = 0
                    ln_ = len(batchesTrain)
                    # batchesTrain = batchesTrain[:50]
                    for nextBatch in tqdm(batchesTrain, desc="Training"):
                    # input
                        feed = {
                                model.input_x_word: nextBatch.input_x_word,
                                model.input_mask: nextBatch.input_mask,
                                model.input_relation: nextBatch.inputs_y,
                                model.input_x_len: nextBatch.sequence_lengths,
                                model.keep_prob: args.keep_prob,
                                model.is_training: True
                                }
                        _, step, _, loss, lr = session.run(
                                fetches=[train_op,
                                        global_step,
                                        embed_step,
                                        model.loss,
                                        learning_rate
                                        ],
                                feed_dict=feed)
                        current_step += 1
                        loss_ += loss
                    
                    tqdm.write("----- Step %d -- train:  '\n'loss %.6f " % ( current_step, loss_/ln_))
                    
                    P, R, F1 = getScore(model, dev_data, session, args, tokenizer)
                    print('dev set : precision: {:.6f}, recall {:.6f}, f1 {:.6f}\n'.format(P, R, F1))
                    
                    """record"""
                    log_txt = open(log_dir,"a+")
                    print(str(datetime.now())[0:16],
                        'dev_set: precision: {:.6f}, recall {:.6f}, f1 {:.6f}\n'.format(P, R, F1),
                        file=log_txt)
                    log_txt.close()
                    
                    """save model"""
                    if F1 > args.f1_threshold:                                   
                        out_dir = os.path.join(args.model_saved_root, "{}".format(mode))
                        print('check_path==>',f'out_dir: {out_dir}')

                        if not os.path.exists(out_dir):
                            os.makedirs(out_dir)
                        
                        P_ = str(format(P,'.4f'))[2:]
                        R_ = str(format(R,'.4f'))[2:]
                        F1_ = str(format(F1,'.4f'))[2:]
                        model_name = 'model_'+'P'+P_+'-'+'R'+R_+'-'+'F1'+F1_
                        # print('model_name==>',model_name)
                        model_saved_name = os.path.join(out_dir, model_name)
                        scores, model_names, flag_renew = scores_renew(scores, model_names, F1, model_saved_name)
                        
                        if flag_renew:
                            # os.system(f"rm -f {model_names[-1]}*")
                            print('model_name==>',model_name)
                        saver.save(sess=session, 
                                save_path=model_saved_name, 
                                global_step=step)

                        log_txt = open(log_dir,"a+")
                        print(f'model_saved_name: {model_saved_name}',
                        file=log_txt)
                        log_txt.close()


def delete_model(args):
    Model_saved = args.model_saved_root #model_saved
    for fold_idx in range(args.k_folds):
        model_root = os.path.join(Model_saved,str(fold_idx)) #model_saved/0
        print(f"model_root: {model_root}")
        if not os.path.exists(model_root):
            pass
        for root, dirs, files in os.walk(model_root,topdown=False):
            f1_list = []
            file_dict = {}
            all_files = []
            for every_file in files:
                all_files.append(every_file)
                if every_file.endswith('meta'):
                    f1_score = "0."+every_file.split('-')[-2][2:]
                    # print(f"f1_score: {f1_score}")
                    f1_list.append(f1_score)
                    model_name = every_file[:-5]
                    file_dict[every_file] = f1_score
                    # print(f'file_dict:{file_dict}')
                else:
                    continue
        f1_sequence = sorted(f1_list,reverse=True)
        # print(f"f1_sequence: {f1_sequence}")
        f1_max = f1_sequence[0]
        # print(f'file_dict:{file_dict}')
        new_dict =  {v : k for k, v in file_dict.items()}
        # print(f'new_dict:{new_dict}')
        model_selected = new_dict[f1_max]
        # print("model_selected",model_selected)
        model_selected_name = model_selected.split('.')[0]
        # print('all_files',all_files)

        for i in all_files:
            if i=="checkpoint":
                continue
            elif  i.split('.')[0] == model_selected_name:
                pass
            else:
                print(os.path.join(str(fold_idx),i))
                delete_file = Model_saved + '/' + os.path.join(str(fold_idx),i)
                os.remove(delete_file)   



if __name__=='__main__':
    """parse arguments"""
    args = cfg.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    print(f'assign gpu={os.environ["CUDA_VISIBLE_DEVICES"]}')

    """ create experiment file """ 
    today = str(datetime.now())[5:10]
    experiment_name =today + '_' + args.experiment_name
    experiment_path =  os.path.join(args.experiment_root, experiment_name)
    if not os.path.exists(experiment_path):
        os.mkdir(experiment_path)
    print(f'experiment_path:{experiment_path}')
    
    """create model saved dir"""
    model_saved_root = os.path.join(experiment_path,args.model_saved)
    print(f"model_saved_path:{model_saved_root}")
    if not os.path.exists(model_saved_root):
        os.mkdir(model_saved_root)
    # add in  args
    args.model_saved_root = model_saved_root
     
    """ create log file """
    log_name = experiment_name + '.txt'
    log_dir = experiment_path + '/' + log_name
    print('log_dir=',log_dir)
    
    """ experiment record """
    cfg.record_args(log_dir, args)

    """bert"""
    args.vocab_file = args.bert_root + 'vocab.txt'
    args.bert_config_file = args.bert_root + 'bert_config.json'
    args.bert_file = args.bert_root + 'bert_model.ckpt'

    """train"""
    Train(args)

    """delete non-best model in every fold"""
    delete_model(args)

    print(datetime.now(), 'Train Finished!')