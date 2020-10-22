#!/usr/bin/env python
# -*- coding: utf-8 -*-
# /home/celia/Traditional_Chinese_Medicine_MyProject/Utils/data_utils.py
""" function: read data """
import pandas as pd
from tqdm import tqdm
import os
import numpy as np
import json
from Cfg_files import cfg

# import sys
# sys.path.insert(0,'./Config')
# import Config
# from Config.default import update_config
# # from config_files import cfg_test, cfg_train

args = cfg.parse_args()
baseTrainPath = args.baseTrainPath
baseTestPath = args.baseTestPath

def readTxt(path, basePath):
    path = basePath + path
    file = open(path, 'r', encoding='utf8').readlines()
    if len(file)!=1:
        return ''
    else:
        return file[0]

def readLabel(path, iid):
    path = baseTrainPath + path
    # print('check data path==>',path) # read label from every ann
    dta = pd.read_csv(path, sep='\t', names=['id','entityInfo','entity'])
    # print(dta)
    dta['category'] = dta['entityInfo'].apply(lambda x:x.split(' ')[0])
    dta['pe1'] = dta['entityInfo'].apply(lambda x:x.split(' ')[1]).astype(int)
    dta['pe2'] = dta['entityInfo'].apply(lambda x:x.split(' ')[2]).astype(int)
    dta['id'] = iid
    dta = dta[['id','entity','category','pe1','pe2']]
    return dta

def load_data(data_dir):
    df = {}
    dfLabel = pd.DataFrame()

    for path in tqdm(os.listdir(data_dir)):
        cid = path[-3:]
        if cid=='txt':
            df[path[:-4]] = readTxt(path, data_dir)
        else:
            dta = readLabel(path, path[:-4])
            dfLabel = pd.concat( [dfLabel, dta] )
    # print("\n")
    # print(set(dfLabel['entity']))
    print ( '\n ... 训练集实体个数为{0}，实体类别个数为{1} ...'.format( 
            len(set(dfLabel['entity'])), len(set(dfLabel['category'])) ) )
    # 训练集实体个数为3417，实体类别个数为13
    # 训练集实体个数为3783，实体类别个数为13 # 10.21
    
    # print_file = "v2_data_format.txt"
    # a = open(print_file,"a+")
    # print(f"df: {df}\n",f"\ndfLable: {dfLabel}\n",file=a)
    # a.close()
    
    return df, dfLabel

def load_testData():
    test = {}
    for path in tqdm(os.listdir( baseTestPath)):
        test[path[:-4]] = readTxt(path, baseTestPath)
    return test

def read_json_from_file(file_name):
    print('loading ... ' + file_name)
    fp = open(file_name, "rb")
    data = json.load(fp)
    fp.close()
    return data

def initLabel():
    # print(f"args.categoryLable:{args.categoryLable}")
    if os.path.exists(args.categoryLable):
        category2id_1 = read_json_from_file(args.categoryLable)
    else:
        print ('error-category2id')
    # print("what's category2id_1 like?==>",type(category2id_1)) #dict
    return category2id_1

def trans2data(dta, category2id, max_x_length):
    # print(f"dta['label']:{dta['label']}")
    dta['y'] = dta['label'].apply(lambda x: [ category2id[i] for i in x ])
    dta_ = []
    for d in tqdm(dta.iterrows()):
        d = d[1]
        text = d['text'][:max_x_length]
        lnt = len(text)
        dta_.append( [ text, lnt, d['y'], d['l_other'] ] )
    np.random.shuffle(dta_)
    
    # """see see"""
    # print_txt = open(args.print_sth,"a+")
    # print('===from trans2data_train===\n', dta_, '\n',
    #     file=print_txt)
    # print_txt.close()

    return dta_

def data_preprocess():
    category2id = initLabel() #category mapping
    id2category = {j:i for i,j in category2id.items()}
    return category2id, id2category

def ssbs( df, dfLabel, maxLen ):
    ''' sample split by Symbol '''

    def _split(iid, df, dfLabel):
        text = df[str(iid)]
        lb = dfLabel[dfLabel['id'] == str(iid)]
        lines = text.split('。')
        
        # 记录分词后的每个短句对应的起止位置
        infos = []
        for offset, tx in enumerate( lines ):
            if offset==0:
                info = { 'offset':offset, 
                               'lineStartPosition':0, 
                               'lineEndPosition':len(tx)-1 }
                lastEndPos = info['lineEndPosition']
            else:
                info = { 'offset':offset, 
                               'lineStartPosition':lastEndPos+1+1, 
                               'lineEndPosition':lastEndPos+1+1+len(tx)-1  }
                lastEndPos = info['lineEndPosition']
            
            entityNum = len (lb[ (lb['pe1']>=info['lineStartPosition']) &
                          (lb['pe2']<=info['lineEndPosition']) ] )
            info['entityNum'] = entityNum
            infos.append( info )
        infos = pd.DataFrame(infos)[['offset', 'entityNum', 'lineStartPosition', 'lineEndPosition']]
        
        ''' infos-->
                    offset  entityNum  lineStartPosition  lineEndPosition
                    0         3                  0               46
                    1         1                 48               58
                    2        16                 60              183
                    3         0                185              218
                    4         0                220              236
                    5         0                238              238       '''
        
        # 长句拆分为短句
        def oneSample( lineStartPosition, dta):
            dta_ = dta[dta['lineEndPosition'] >= maxLen+lineStartPosition]
            dta = dta[dta['lineEndPosition'] < maxLen+lineStartPosition]
            if len(dta)==0:
                return rs
            rs.append( {'offsetStart':dta['offset'].values[0],
             'offsetEnd':dta['offset'].values[-1],
             'lineStartPosition':dta['lineStartPosition'].values[0],
             'lineEndPosition':dta['lineEndPosition'].values[-1]} )
            
            lineStartPosition = dta['lineEndPosition'].values[-1] + 2
            oneSample( lineStartPosition, dta_)
            return rs
        
        rs = []
        lineStartPosition = 0
        rs = oneSample( lineStartPosition, infos.copy())
        
        #构造训练样本  
        #=################## 如果用不同标注方式 要重新生成json文件
        samples = []
        for s in rs:
            # 拆分为短句后的句子
            tx = text[ s['lineStartPosition']:s['lineEndPosition']+1 ]
            label = lb[ (lb['pe1']>=s['lineStartPosition']) &
                          (lb['pe2']<=s['lineEndPosition']) ]
            l_ = ['O'] * len(tx) # 训练的label
            l_other = []
            for left,right,cate,entity in zip(label['pe1'],label['pe2'],label['category'],label['entity']):
                left = left-s['lineStartPosition']
                right = right-s['lineStartPosition']
                l_other.append( (left,right,cate,entity) )
                try:
                    if left == right:
                        l_[left] = 'S'+'-'+cate
                        
                    else:
                        l_[left] = 'B'+'-'+cate
                        for i in range(left+1,right-1):
                            l_[i] = 'I'
                        l_[right-1] = 'I'
                except:
                    print(f"error occur in : \n{label} \n{left}")
            samples.append( {'text':tx, 'label':l_, 'l_other':l_other} )
            
        return pd.DataFrame(samples)
    
    
    train = pd.DataFrame()
    for iid in df.keys():
        train = pd.concat( [train, _split(iid, df, dfLabel)] )
    
    # """see see"""
    # print_txt = open(args.print_sth,"a+")
    # print('===from ssbs_train===\n', train, '\n',
    #     file=print_txt)
    # print_txt.close()
    
    return train




def ssbsTest( df, maxLen ):
    ''' sample split by Symbol '''
    def _split( text ):
        lines = text.split('。')

        # 记录分词后的每个短句对应的起止位置
        infos = []
        for offset, tx in enumerate( lines ):
            if offset==0:
                info = { 'offset':offset, 
                               'lineStartPosition':0, 
                               'lineEndPosition':len(tx)-1 }
                lastEndPos = info['lineEndPosition']
            else:
                info = { 'offset':offset, 
                               'lineStartPosition':lastEndPos+1+1, 
                               'lineEndPosition':lastEndPos+1+1+len(tx)-1  }
                lastEndPos = info['lineEndPosition']
            infos.append( info )
        infos = pd.DataFrame(infos)[['offset', 'lineStartPosition', 'lineEndPosition']]
        
        # 长句拆分为短句
        def oneSample( lineStartPosition, dta):
            
            dta_ = dta[dta['lineEndPosition']>=maxLen+lineStartPosition]
            dta = dta[dta['lineEndPosition']<maxLen+lineStartPosition]
            if len(dta)==0:
                return rs
            rs.append( {'offsetStart':dta['offset'].values[0],
             'offsetEnd':dta['offset'].values[-1],
             'lineStartPosition':dta['lineStartPosition'].values[0],
             'lineEndPosition':dta['lineEndPosition'].values[-1]} )
            
            lineStartPosition = dta['lineEndPosition'].values[-1] + 2
            oneSample( lineStartPosition, dta_)
            return rs

        rs = []
        lineStartPosition = 0
        rs = oneSample( lineStartPosition, infos.copy())
        
        #构造训练样本
        samples = []
        for s in rs:
            # 拆分为短句后的句子
            tx = text[ s['lineStartPosition']:s['lineEndPosition']+1 ]
            s['tx'] = tx
            s['id'] = iid
            s['text'] = text
            samples.append( s )
            
        return pd.DataFrame(samples)
    
    test = pd.DataFrame()
    for iid in df.keys():
        test = pd.concat( [test, _split( df[str(iid)] )] )
    
    return test



