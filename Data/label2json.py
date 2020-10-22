import json
import os
import sys 
# sys.path.insert(0, "..") 
# print('===>',os.getcwd())

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from Utils import data_utils
#from Cfg_files import cfg

train_data = ""
max_len = 500

def save_json_to_file(file_name,data):
    print('save .. ' + file_name)
    fp = open(file_name,"w")
    json.dump(data,fp)
    fp.close()

total_data, dfLabel = data_utils.load_data(train_data)
print_txt = open("label2json_dfLabel.txt","w")
print('==dfLabel===\n', dfLabel, file=print_txt)
print_txt.close() 

train = data_utils.ssbs( total_data, dfLabel, max_len)




print_txt = open("label2json_after_ssbs.txt","w")
print('==dev_data after ssbs===\n', train[1:10], file=print_txt)
print_txt.close()


words = set( [i for l in train['label'] for i in l] ) | set(["[CLS]", "[SEP]"])
category2id = {w: idx for idx, w in enumerate( words )}
save_json_to_file('./category2id.json',category2id)

