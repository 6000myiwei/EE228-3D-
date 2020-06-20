import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from tensorflow.python.keras.models import load_model
import tensorflow as tf
import pandas as pd
import numpy as np

from mylib.models import metrics, losses, densenet_max, densenetf,resnet,densenet,densenetf_avr

physical_devices = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

'''
测试数据指定：
'''
data_path = './test'

'''
模型路径指定：
'''
model_path = ['./model/0.688(224attentionmaxpoolingfk=18)/weights.150.h5',
              './model/0.686(224attentionmaxpoolingfbottle)/weights.120.h5',]


crop_size=[32, 32, 32]
model = []

model.append(densenetf.get_model(down_structure=[2,2,4],k=18,weights=model_path[0]))
model.append(densenetf.get_model(down_structure=[2,2,4],k=16,weights=model_path[1]))



from dataloader import ClfAttentionDataset,get_test_loader
lines = pd.read_csv('test.csv')
pred = []
for m in model:
    test_dataset = ClfAttentionDataset(crop_size=crop_size, subset=['test'], move=None,lines=lines,data_path=data_path)
    test_loader = get_test_loader(test_dataset, batch_size=1)
    pred.append(m.predict(test_loader,steps=len(test_dataset)))
#%%
total = np.zeros(117)

for i in range(len(model_path)):
    total += pred[i][:,1].squeeze()
    
predicted = total / len(model_path)
candidate = lines['name'].tolist()
result = {'name': candidate, 'predicted':predicted}

result = pd.DataFrame(result)
result.to_csv("submission.csv",index=False,sep=',')