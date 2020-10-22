import argparse
from datetime import datetime
import os 
save_pth = ""
if not os.path.exists(save_pth):
    os.makedirs(save_pth)

def parse_args():
    parser = argparse.ArgumentParser(description='parameters configuration.')
    parser.add_argument('--fold_id', type=int, required=True) # be used to train one fold in one GPU.
    
    #general
    parser.add_argument('--gpu_id', type=str, required=True)
    # parser.add_argument('--gpu_id', type=str, required=False, default="0")
    parser.add_argument('--experiment_name', type=str, required=True) 
    parser.add_argument('--experiment_root', type=str, default=save_pth)
    parser.add_argument('--max_x_length', help='max_x_length', type=int, default=500)
    parser.add_argument('--bert_root',help='path of pretrained bert_model', type=str, default='./Bert/publish/chinese_wwm_ext_L-12_H-768_A-12/')
    # parser.add_argument('--bert_root',help='path of pretrained bert_model', type=str, default='./Bert/publish/wwm_cased_L-24_H-1024_A-16/')

    parser.add_argument('--print_sth', help="see see what's this?", type=str, default='print.txt')
    parser.add_argument('--baseTrainPath', type=str, default='')
    parser.add_argument('--baseTestPath', type=str, default='')
    parser.add_argument('--categoryLable', type=str, default='./Data/cache/category2id.json')
    parser.add_argument('--devided_data', help="train&dev data", type=str, default='./Data/cache/RandomDataOrder_TrainVal_fixed.json')
    parser.add_argument('--mode', help="train or test", type=str, default='train') #in test,can be modified automatically
    
    ##model
    parser.add_argument('--model_type',help='choice: [idcnn, bilstm]', type=str, default='bilstm')#
    parser.add_argument('--lstm_dim',help='lstm_dim ->256 is better', type=int, default=256) 
    parser.add_argument('--use_origin_bert', help='use_origin_bert or dynamic_fusion_bert,default:latter', type=bool, default=False)            
    parser.add_argument('--embed_dense', type=bool, default=True)
    parser.add_argument('--embed_dense_dim', type=int, default=1024)  #1024 is better
    parser.add_argument('--use_bert', type=bool, default=True)
    parser.add_argument('--num_layer', type=int, default=5) #5 is enough
    

    ##train
    parser.add_argument('--train_epoch', type=int, default=20)
    parser.add_argument('--learning_rate', type=float, default=1e-4)#1e-4
    parser.add_argument('--batch_size', type=int, default=7) #in one gpu, max=8 2080Ti
    parser.add_argument('--k_folds', type=int, default=5)
    parser.add_argument('--f1_threshold', help='f1_threshold for save model', type=float, default=0.6)
    parser.add_argument('--dropout', type=float, default=0.85)
    parser.add_argument('--warmup_propotion',help='a parameter in bert', type=float, default=0.05)
    parser.add_argument('--decay_rate', type=float, default=0.85)
    parser.add_argument('--decay_step', type=float, default=200)
    parser.add_argument('--keep_prob', type=float, default=0.9)
    parser.add_argument('--continue_training', type=bool, default=False)
    parser.add_argument('--checkpointPath', type=str, default="")
    parser.add_argument('--diceloss_weight', type=float, default=0)
    parser.add_argument('--save_ckpt_num', help='save_ckpt_num', type=int, default=2)

    parser.add_argument('--embed_learning_rate', type=float, default=5e-5)
    parser.add_argument('--num_checkpoints', help='a param in tf.train.Saver', type=int, default=3)
    parser.add_argument('--model_saved', type=str, default='model_saved')
    
    ##test
    parser.add_argument('--alpha', help='hyper-parameter in NewtonCoolingLaw', type=float, default=1/4)
    parser.add_argument('--model_to_test', help='path of model to test ', type=str, default=''")
    parser.add_argument('--pred_results', help='path to save test result', type=str, default="pred_results")
    parser.add_argument('--num_model_merge', help='num_model_merge', type=int, default=7)
                        
    args = parser.parse_args()

    return args                   


def record_args(log_file, args):
    if args.mode == 'train':
        logger = open(log_file,"a+")
        print(datetime.now(), f" experiment record==>",'\n',
            f"general==>",'\n',
                f"gpu_id: {args.gpu_id}\n",
                f"fold_id: {args.fold_id}\n",
                # f"experiment_name: {args.experiment_name}\n",
                f"experiment_root: {args.experiment_root}\n",
                f"max_x_length: {args.max_x_length}\n",
                f"bert_root: {args.bert_root}\n",
                f"baseTrainPath: {args.baseTrainPath}\n",
                f"baseTestPath: {args.baseTestPath}\n",
                f"categoryLable: {args.categoryLable}\n",
                # f"devided_data: {args.devided_data}\n",
            f"model==>",'\n',
                f"model_type: {args.model_type}\n",
                f"lstm_dim: {args.lstm_dim}\n",
                f"use_origin_bert: {args.use_origin_bert}\n",
                f"embed_dense: {args.embed_dense}\n",
                f"embed_dense_dim: {args.embed_dense_dim}\n",
                f"use_bert: {args.use_bert}\n",
                f"num_layer: {args.num_layer}\n",
            f"train==>",'\n',
                f"train_epoch: {args.train_epoch}\n",
                f"learning_rate: {args.learning_rate}\n",
                f"batch_size: {args.batch_size}\n",
                f"k_folds: {args.k_folds}\n",
                f"f1_threshold: {args.f1_threshold}\n",
                f"dropout: {args.dropout}\n",
                f"warmup_propotion: {args.warmup_propotion}\n",
                f"decay_rate: {args.decay_rate}\n",
                f"decay_step: {args.decay_step}\n",
                f"keep_prob: {args.keep_prob}\n",
                f"f1_threshold: {args.f1_threshold}\n",
                f"continue_training: {args.continue_training}\n",
                f"embed_learning_rate: {args.embed_learning_rate}\n",
                f"num_checkpoints: {args.num_checkpoints}\n",
                f"model_saved: {args.model_saved}\n",
                f"save_ckpt_num: {args.save_ckpt_num}\n",
                f"num_model_merge: {args.num_model_merge}\n",
                f"diceloss_weight: {args.diceloss_weight}\n",
                

                file=logger)
        logger.close()
    
    else:
        logger = open(log_file,"a+")
        print(datetime.now(), '\n',
            f"test ==>",'\n',
                f"gpu_id: {args.gpu_id}\n",
                f"model_saved: {args.model_saved}\n",
                f"alpha: {args.alpha}\n",
                f"f1_threshold: {args.f1_threshold}\n",
                f"model_to_test: {args.model_to_test}\n",
                f"pred_results: {args.pred_results}\n",

                file=logger)  
        logger.close()    
                        
                        
                        
                        
                        
                       
                        
                                          
                        
                        
                        
                       
                        
                        
                       
                        
                        
                                        
                        
                        
                        
    
    
    
    
