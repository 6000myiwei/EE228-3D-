# -*- coding: utf-8 -*-
"""
Created on Wed Jun 17 09:57:32 2020

@author: 6000myiwei
"""

from tensorflow.python.keras.layers import (Conv3D, BatchNormalization, AveragePooling3D,MaxPooling3D, concatenate, Lambda,
                          Activation, Input, GlobalAvgPool3D, Dense,SpatialDropout3D,add,ZeroPadding3D)
from tensorflow.python.keras.regularizers import l2 as l2_penalty
from tensorflow.python.keras.models import Model

from mylib.models.metrics import invasion_acc, invasion_precision, invasion_recall, invasion_fmeasure
import keras.backend as K
import tensorflow as tf

PARAMS = {
    'activation': lambda: Activation('relu'),  # the activation functions
    'bn_scale': True,  # whether to use the scale function in BN
    'weight_decay': 1e-6,  # l2 weight decay
    'kernel_initializer': 'he_uniform',  # initialization
    'first_scale': lambda x: x / 128. - 1.,  # the first pre-processing function
    'dhw': [32, 32, 32],  # the input shape
    'output_size': 2,  # the output number of the classification head
    'dropout_rate': None  # whether to use dropout, and how much to use
}

def Conv3d_BN(x, nb_filter, kernel_size, strides=(1, 1, 1), padding='same', name=None):
    bn_scale = PARAMS['bn_scale']
    kernel_initializer = PARAMS['kernel_initializer']
    weight_decay = PARAMS['weight_decay']

    if name is not None:
        bn_name = name + '_bn'
        conv_name = name + '_conv'
    else:
        bn_name = None
        conv_name = None
 
    x = Conv3D(nb_filter, kernel_size, padding=padding, strides=strides, 
               activation='relu', name=conv_name,
               kernel_initializer=kernel_initializer,
               kernel_regularizer=l2_penalty(weight_decay))(x)
    x = BatchNormalization(axis=3, name=bn_name,scale=bn_scale)(x)
    return x
 
 
def identity_Block(inpt, nb_filter, kernel_size, strides=(1, 1, 1), with_conv_shortcut=False):
    x = Conv3d_BN(inpt, nb_filter=nb_filter, kernel_size=kernel_size, strides=strides, padding='same')
    x = Conv3d_BN(x, nb_filter=nb_filter, kernel_size=kernel_size, padding='same')
    if with_conv_shortcut:
        shortcut = Conv3d_BN(inpt, nb_filter=nb_filter, strides=strides, kernel_size=kernel_size)
        x = add([x, shortcut])
        return x
    else:
        x = add([x, inpt])
        return x

def get_model(weights):
    
    dhw = PARAMS['dhw']
    
    classes = PARAMS['output_size']
    
    shape = dhw + [1]
    
    inputs = Input(shape)
    
    first_scale = PARAMS['first_scale']
    
    if first_scale is not None:
        scaled = Lambda(first_scale)(inputs)
    else:
        scaled = inputs
        
    x = ZeroPadding3D((3, 3, 3))(inputs)
 
    #conv1
    x = Conv3d_BN(x, nb_filter=16, kernel_size=(7, 7, 7), strides=(2, 2, 2), padding='valid')
    x = MaxPooling3D(pool_size=(3, 3, 3), strides=(2, 2, 2), padding='same')(x)
 
    #conv2_x
    x = identity_Block(x, nb_filter=16, kernel_size=(3, 3, 3))
    x = identity_Block(x, nb_filter=16, kernel_size=(3, 3, 3))
 
    #conv3_x
    x = identity_Block(x, nb_filter=32, kernel_size=(3, 3, 3), strides=(2, 2 ,2), with_conv_shortcut=True)
    x = identity_Block(x, nb_filter=32, kernel_size=(3, 3, 3))
 
    #conv4_x
    x = identity_Block(x, nb_filter=64, kernel_size=(3, 3, 3), strides=(2, 2, 2), with_conv_shortcut=True)
    x = identity_Block(x, nb_filter=64, kernel_size=(3, 3, 3))
 
    # #conv5_x
    # x = identity_Block(x, nb_filter=512, kernel_size=(3, 3), strides=(2, 2), with_conv_shortcut=True)
    # x = identity_Block(x, nb_filter=512, kernel_size=(3, 3))
    x = GlobalAvgPool3D()(x)
    x = Dense(classes, activation='softmax')(x)
 
    model = Model(inputs=inputs, outputs=x)
    model.summary()
    
    if weights is not None:
        model.load_weights(weights)
        print('load weights:', weights)
    return model

def get_compiled(loss='categorical_crossentropy', optimizer='adam',
                 metrics=["categorical_accuracy", invasion_acc,
                          invasion_precision, invasion_recall, invasion_fmeasure],
                 weights=None, **kwargs):
    model = get_model(weights=weights)
    model.compile(loss=loss, optimizer=optimizer,
                  metrics=[loss] + metrics)
    return model

if __name__ == '__main__':
    model = get_compiled()