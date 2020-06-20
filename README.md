# EE228 course project M3DV  
model文件夹：训练的好的模型参数文件，共两个，总计1.97MB  

mylib文件夹：包括构建模型，数据增强，评价指标等代码，参考自https://github.com/duducheng/DenseSharp  

dataloader.py:加载数据的生成器  

划分验证集.py:将数据集分成训练集和验证集，10折交叉验证  

train.ipynb:训练时所用的notebook文件  

test.py:实现对测试集的测试，可以指定测试集的路径和模型参数的路径  
>>data_path:指定测试集路径，代码中设定的为'./test'  
>>model_path:指定模型路径，代码中设定的为['./model/0.688(224attentionmaxpoolingfk=18)/weights.150.h5',
                                         './model/0.686(224attentionmaxpoolingfbottle)/weights.120.h5']

