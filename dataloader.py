# -*- coding: utf-8 -*-
"""
Created on Fri May 15 10:37:56 2020

@author: HP
"""


import random
import os
import numpy as np
import pandas as pd

from collections.abc import Sequence
import tensorflow as tf

from mylib.utils.misc import rotation, reflection, crop, random_center, _triple
# def gen():
#     lines = pd.read_csv('train_val.csv')
    
#     index = 0
#     while True:
#         data = np.load('./train_val/' + lines['name'][index] + '.npz')
#         voxel = data['voxel']
#         seg = data['seg']
#         label = lines['label'][index]
#         yield (voxel, seg, label)
#         index += 1
#         if index == len(lines):
#             index = 0





# '''INFO csv文件相当于lines'''

LABEL = [0, 1]
lines = pd.read_csv('train_10fold.csv')


def label_smoothing(inputs, epsilon=0.05):

        K = inputs.shape[-1]
        return ((1-epsilon) * inputs) + (epsilon / K)



class ClfDataset(Sequence):
    def __init__(self, crop_size=32, move=3, subset=[0,1,2,3],lines=lines,
                 define_label = label_smoothing,data_path='./train_val'):
        '''The classification-only dataset.
        :param crop_size: the input size
        :param move: the random move
        :param subset: choose which subset to use
        :param define_label: how to define the label. default: for 3-output classification one hot encoding.
        '''
        self.lines = lines
        self.data_path = data_path
        index = []
        for sset in subset:
            index += list(self.lines[self.lines['subset'] == sset].index)
        self.index = tuple(sorted(index))  # the index in the info
        self.label = np.array([[int(label == s) for label in LABEL] for s in self.lines.loc[self.index, 'label']])
        # self.label = np.array([s for s in self.lines.loc[self.index, 'label']])
        self.transform = Transform(crop_size, move)
        self.define_label = define_label

    def __getitem__(self, item):
        name = self.lines.loc[self.index[item], 'name']
        with np.load(os.path.join(self.data_path, '%s.npz' % name)) as npz:
            voxel = self.transform(npz['voxel'])
        label = self.define_label(self.label[item])
        return voxel, label

    def __len__(self):
        return len(self.index)

    @staticmethod
    def _collate_fn(data):
        xs = []
        ys = []
        for x, y in data:
            xs.append(x)
            ys.append(y)
        return np.array(xs), np.array(ys)


class ClfAttentionDataset(ClfDataset):
    '''Classification and segmentation dataset.'''

    def __getitem__(self, item):
        name = self.lines.loc[self.index[item], 'name']
        with np.load(os.path.join(self.data_path, '%s.npz' % name)) as npz:
            voxel = self.transform(npz['voxel'] * (npz['seg'] * 0.3 + 0.7))
        label = self.define_label(self.label[item])
        return voxel, label

    @staticmethod
    def _collate_fn(data):
        xs = []
        ys = []
        for x, y in data:
            xs.append(x)
            ys.append(y)
        return np.array(xs), np.array(ys)


class ClfSegDataset(ClfDataset):
    '''Classification and segmentation dataset.'''

    def __getitem__(self, item):
        name = self.lines.loc[self.index[item], 'name']
        with np.load(os.path.join(self.data_path, '%s.npz' % name)) as npz:
            voxel, seg = self.transform(npz['voxel'], npz['seg'])
            # voxel = self.transform(npz['voxel'] * (npz['seg'] * 0.8 + 0.2))
        label = self.define_label(self.label[item])
        return voxel, (label, seg)

    @staticmethod
    def _collate_fn(data):
        xs = []
        ys = []
        segs = []
        for x, y in data:
            xs.append(x)
            ys.append(y[0])
            segs.append(y[1])
        return np.array(xs), {"clf": np.array(ys), "seg": np.array(segs)}
        # return tf.cast(np.array(xs),tf.float32), {"clf": np.array(ys), "seg": np.array(segs)}


def get_test_loader(dataset, batch_size):
    total_size = len(dataset)
    print('Size', total_size)
    index_generator = iter(range(total_size))
    while True:
        data = []
        for _ in range(batch_size):
            idx = next(index_generator)
            data.append(dataset[idx])
        yield dataset._collate_fn(data)

def get_loader(dataset, batch_size):
    total_size = len(dataset)
    print('Size', total_size)
    index_generator = shuffle_iterator(range(total_size))
    while True:
        data = []
        for _ in range(batch_size):
            idx = next(index_generator)
            data.append(dataset[idx])
        yield dataset._collate_fn(data)

def get_balanced_loader(dataset, batch_sizes):
    assert len(batch_sizes) == len(LABEL)
    total_size = len(dataset)
    print('Size', total_size)
    index_generators = []
    for l_idx in range(len(batch_sizes)):
        # this must be list, or `l_idx` will not be eval
        iterator = [i for i in range(total_size) if dataset.label[i, l_idx]]
        index_generators.append(shuffle_iterator(iterator))
    while True:
        data = []
        for i, batch_size in enumerate(batch_sizes):
            generator = index_generators[i]
            for _ in range(batch_size):
                idx = next(generator)
                data.append(dataset[idx])
        yield dataset._collate_fn(data)
        
def get_mixup_gen(g1,g2,alpha,batch_size):
    index1 = np.arange(batch_size)
    index2 = np.arange(batch_size)
    
    while True:
        l = np.random.beta(alpha, alpha, batch_size)
        x = next(g1)
        y = next(g2)
        random.shuffle(index1)
        random.shuffle(index2)
        data = x[0][index1] * l.reshape(batch_size, 1, 1, 1, 1) +  \
            y[0][index2] * (1 - l.reshape(batch_size,1 ,1 ,1 ,1))
        
        label = x[1][index1] * l.reshape(batch_size, 1) + \
            y[1][index2] * (1 - l.reshape(batch_size, 1))
        
        yield data,label
        
            
        
        
        
    
    
        
class MixupGenerator():
    def __init__(self, dataset,batch_sizes,alpha=0.2):
        """Constructor for mixup image data generator.
        Arguments:
        generator {object} -- An instance of Keras ImageDataGenerator.
        directory {str} -- Image directory.
        batch_size {int} -- Batch size.
        img_height {int} -- Image height in pixels.
        img_width {int} -- Image width in pixels.
        Keyword Arguments:
        alpha {float} -- Mixup beta distribution alpha parameter. (default: {0.2})
        subset {str} -- 'training' or 'validation' if validation_split is specified in
        `generator` (ImageDataGenerator).(default: {None})
        """
        self.batch_index = 0
        self.batch_size = sum(batch_sizes)
        self.alpha = alpha
        # First iterator yielding tuples of (x, y)
        self.generator1 = get_balanced_loader(dataset, batch_sizes=batch_sizes)
        # Second iterator yielding tuples of (x, y)
        self.generator2 = get_balanced_loader(dataset, batch_sizes=batch_sizes)
        # Number of images across all classes in image directory.
        self.n = len(dataset)
    def reset_index(self):
        """
        Reset the generator indexes array.
        """
        self.generator1._set_index_array()
        self.generator2._set_index_array()
   
    def on_epoch_end(self):
        self.reset_index()
     
    def reset(self):
        self.batch_index = 0
     
    def __len__(self):
         # round up
         return (self.n + self.batch_size - 1) // self.batch_size
     
    def get_steps_per_epoch(self):
        """
        Get number of steps per epoch based on batch size and
        number of images.
        Returns:
        int -- steps per epoch.
        """
        return self.n // self.batch_size
       
    def __next__(self):
        """
        Get next batch input/output pair.
        Returns:
        tuple -- batch of input/output pair, (inputs, outputs).
        """
        if self.batch_index == 0:
            self.reset_index()
        current_index = (self.batch_index * self.batch_size) % self.n
        if self.n > current_index + self.batch_size:
            self.batch_index += 1
        else:
           self.batch_index = 0
           # random sample the lambda value from beta distribution.
        l = np.random.beta(self.alpha, self.alpha, self.batch_size)
        X_l = l.reshape(self.batch_size, 1, 1, 1)
        y_l = l.reshape(self.batch_size, 1)
        # Get a pair of inputs and outputs from two iterators.
        X1, y1 = self.generator1.next()
        X2, y2 = self.generator2.next()
        # Perform the mixup.
        X = X1 * X_l + X2 * (1 - X_l)
        y = y1 * y_l + y2 * (1 - y_l)
        return X, y
    def __iter__(self):
        while True:
            yield next(self)




class Transform:
    '''The online data augmentation, including:
    1) random move the center by `move`
    2) rotation 90 degrees increments
    3) reflection in any axis
    '''

    def __init__(self, size, move):
        self.size = _triple(size)
        self.move = move

    def __call__(self, arr, aux=None):
        shape = arr.shape
        if self.move is not None:
            center = random_center(shape, self.move)
            arr_ret = crop(arr, center, self.size)
            angle = np.random.randint(4, size=3)
            arr_ret = rotation(arr_ret, angle=angle)
            # axis = np.random.randint(4) - 1
            # arr_ret = reflection(arr_ret, axis=axis)
            arr_ret = np.expand_dims(arr_ret, axis=-1)
            if aux is not None:
                aux_ret = crop(aux, center, self.size)
                aux_ret = rotation(aux_ret, angle=angle)
                # aux_ret = reflection(aux_ret, axis=axis)
                aux_ret = np.expand_dims(aux_ret, axis=-1)
                return arr_ret, aux_ret
            return arr_ret
        else:
            center = np.array(shape) // 2
            arr_ret = crop(arr, center, self.size)
            arr_ret = np.expand_dims(arr_ret, axis=-1)
            if aux is not None:
                aux_ret = crop(aux, center, self.size)
                aux_ret = np.expand_dims(aux_ret, axis=-1)
                return arr_ret, aux_ret
            return arr_ret


def shuffle_iterator(iterator):
    # iterator should have limited size
    index = list(iterator)
    total_size = len(index)
    i = 0
    random.shuffle(index)
    while True:
        yield index[i]
        i += 1
        if i >= total_size:
            i = 0
            random.shuffle(index)

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    batch_sizes=[2, 2]
    crop_size=[100,100,100]
    train_subset = [0,1,2,3]
    val_subset = [4]
    model_path = None
    random_move=3
    learning_rate=3.e-4
    weight_decay=0.
    save_folder='test'
    epochs=20
    
    batch_size = sum(batch_sizes)

    train_dataset = ClfDataset(crop_size=crop_size, subset=train_subset, move=random_move,lines=lines)

    seg_dataset = ClfSegDataset(crop_size=crop_size, subset=train_subset, move=random_move,lines=lines)
    
    seg_g = get_balanced_loader(seg_dataset, batch_sizes=batch_sizes)

    val_dataset = ClfDataset(crop_size=crop_size, subset=val_subset, move=None,lines=lines)

    train_loader = get_loader(train_dataset, batch_size=1)
    
    val_loader = get_test_loader(val_dataset, batch_size=1)
    
    tmp = next(val_loader)
    voxel = tmp[0].squeeze()
    seg = tmp[1].squeeze()
    
    for i in range(44,50,2):
        plt.figure()
        plt.imshow(voxel[i],cmap='gray')
        plt.show()