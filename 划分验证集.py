# -*- coding: utf-8 -*-
"""
Created on Fri May 15 14:50:39 2020

@author: HP
"""

import pandas as pd
import numpy as np

lines = pd.read_csv('train_val.csv')

lines['subset'] = 'train'

data = lines.values.tolist()
#%%
data = np.array(data)
pos = np.where(data == '1')
neg = np.where(data == '0')
#%%
pos_len = data[pos[0]].shape[0]
neg_len = data[neg[0]].shape[0]
pos_list = list(range(0, pos_len))
neg_list = list(range(0,neg_len))
#%%
from random import shuffle
shuffle(pos_list)
shuffle(neg_list)
#%%
pos = pos[0]
neg = neg[0]

num = 10
stepp = len(pos_list) // 10
stepn = len(neg_list) // 10

for i in range(num):
    cp = pos[pos_list[i*stepp:(i+1)*stepp]]
    cn = neg[neg_list[i*stepn:(i+1)*stepn]]
    if i == num - 1:
        cp = pos[pos_list[i*stepp:]]
        cn = neg[neg_list[i*stepn:]]
    lines.iloc[cp,2] ='%s'%i
    lines.iloc[cn,2] ='%s'%i 
    
#%%
lines.to_csv("train_10fold.csv",index=False,sep=',')