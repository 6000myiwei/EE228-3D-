'''
change bottleneck
maxpooling
'''

from tensorflow.python.keras.layers import (Conv3D, BatchNormalization, AveragePooling3D,MaxPooling3D, concatenate, Lambda,
                          Activation, Input, GlobalAvgPool3D, Dense,SpatialDropout3D,GlobalMaxPooling3D)
from tensorflow.python.keras.regularizers import l2 as l2_penalty
from tensorflow.python.keras.models import Model

from mylib.models.metrics import invasion_acc, invasion_precision, invasion_recall, invasion_fmeasure
import keras.backend as K
import tensorflow as tf

PARAMS = {
    'activation': lambda: Activation('relu'),  # the activation functions
    'bn_scale': True,  # whether to use the scale function in BN
    'weight_decay': 0.,  # l2 weight decay
    'kernel_initializer': 'he_uniform',  # initialization
    'first_scale': lambda x: x / 128. - 1.,  # the first pre-processing function
    'dhw': [32, 32, 32],  # the input shape
    'k': 18,  # the `growth rate` in DenseNet, default 16
    'bottleneck': [2,2,4],  # the `bottleneck` in DenseNet
    'compression': 2,  # the `compression` in DenseNet
    'first_layer': 32,  # the channel of the first layer
    'down_structure': [2,2,4],  # the down-sample structure
    'output_size': 2,  # the output number of the classification head
    'dropout_rate': None  # whether to use dropout, and how much to use
}


def _conv_block(x, filters, bottleneck):
    bn_scale = PARAMS['bn_scale']
    activation = PARAMS['activation']
    kernel_initializer = PARAMS['kernel_initializer']
    weight_decay = PARAMS['weight_decay']
    dropout_rate = PARAMS['dropout_rate']

    x = BatchNormalization(scale=bn_scale, axis=-1)(x)
    x = activation()(x)
    x = Conv3D(filters * bottleneck, kernel_size=(1, 1, 1), padding='same', use_bias=False,
               kernel_initializer=kernel_initializer,
               kernel_regularizer=l2_penalty(weight_decay))(x)
    if dropout_rate is not None:
        x = SpatialDropout3D(dropout_rate)(x)
    
    x = BatchNormalization(scale=bn_scale, axis=-1)(x)
    x = activation()(x)
    x = Conv3D(filters, kernel_size=(3, 3, 3), padding='same', use_bias=True,
               kernel_initializer=kernel_initializer,
               kernel_regularizer=l2_penalty(weight_decay))(x)
    return x


def _dense_block(x, n, bottleneck):
    k = PARAMS['k']

    for _ in range(n):
        conv = _conv_block(x, k, bottleneck)
        x = concatenate([conv, x], axis=-1)
    return x


def _transmit_block(x, is_last):
    bn_scale = PARAMS['bn_scale']
    activation = PARAMS['activation']
    kernel_initializer = PARAMS['kernel_initializer']
    weight_decay = PARAMS['weight_decay']
    compression = PARAMS['compression']

    x = BatchNormalization(scale=bn_scale, axis=-1)(x)
    x = activation()(x)
    if is_last:
        x = GlobalAvgPool3D()(x)
        # x = GlobalMaxPooling3D()(x)
    else:
        *_, f = x.get_shape().as_list()
        x = Conv3D(f // compression, kernel_size=(1, 1, 1), padding='same', use_bias=True,
                   kernel_initializer=kernel_initializer,
                   kernel_regularizer=l2_penalty(weight_decay))(x)
        x = MaxPooling3D((2, 2, 2), padding='valid')(x)
        # x = AveragePooling3D((2, 2, 2), padding='valid')(x)
    return x


def get_model(weights=None, **kwargs):
    for k, v in kwargs.items():
        assert k in PARAMS
        PARAMS[k] = v
    print("Model hyper-parameters:", PARAMS)

    dhw = PARAMS['dhw']
    first_scale = PARAMS['first_scale']
    first_layer = PARAMS['first_layer']
    kernel_initializer = PARAMS['kernel_initializer']
    weight_decay = PARAMS['weight_decay']
    down_structure = PARAMS['down_structure']
    output_size = PARAMS['output_size']
    bottleneck = PARAMS['bottleneck']

    shape = dhw + [1]

    inputs = Input(shape=shape)

    if first_scale is not None:
        scaled = Lambda(first_scale)(inputs)
    else:
        scaled = inputs
    conv = Conv3D(first_layer, kernel_size=(3, 3, 3), padding='same', use_bias=True,
                  kernel_initializer=kernel_initializer,
                  kernel_regularizer=l2_penalty(weight_decay))(scaled)

    downsample_times = len(down_structure)
    for l, n in enumerate(down_structure):
        db = _dense_block(conv, n, bottleneck[l])
        conv = _transmit_block(db, l == downsample_times - 1)

    if output_size == 1:
        last_activation = 'sigmoid'
    else:
        last_activation = 'softmax'

    outputs = Dense(output_size, activation=last_activation,
                    kernel_regularizer=l2_penalty(weight_decay),
                    kernel_initializer=kernel_initializer)(conv)

    model = Model(inputs, outputs)
    model.summary()

    if weights is not None:
        model.load_weights(weights)
        print('load weights:', weights)
    return model


# def binary_focal_loss(y_true, y_pred,gamma=2.0, alpha=0.25):
#     # Define epsilon so that the backpropagation will not result in NaN
#     # for 0 divisor case
#     epsilon = K.epsilon()
#     # Add the epsilon to prediction value
#     #y_pred = y_pred + epsilon
#     # Clip the prediciton value
#     y_pred = K.clip(y_pred, epsilon, 1.0-epsilon)
#     # Calculate p_t
#     p_t = tf.where(K.equal(y_true, 1), y_pred, 1-y_pred)
#     # Calculate alpha_t
#     alpha_factor = K.ones_like(y_true)*alpha
#     alpha_t = tf.where(K.equal(y_true, 1), alpha_factor, 1-alpha_factor)
#     # Calculate cross entropy
#     cross_entropy = -K.log(p_t)
#     weight = alpha_t * K.pow((1-p_t), gamma)
#     # Calculate focal loss
#     loss = weight * cross_entropy
#     # Sum the losses in mini_batch
#     loss = K.sum(loss, axis=1)
#     return loss


# def categorical_squared_hinge(y_true, y_pred):
#     """
#     hinge with 0.5*W^2 ,SVM
#     """
#     y_true = 2. * y_true - 1 # trans [0,1] to [-1,1]，注意这个，svm类别标签是-1和1
#     vvvv = K.maximum(1. - y_true * y_pred, 0.) # hinge loss，参考keras自带的hinge loss
#     vvv = K.square(vvvv) # 文章《Deep Learning using Linear Support Vector Machines》有进行平方
#     vv = K.sum(vvv, 1, keepdims=False)  #axis=len(y_true.get_shape()) - 1
#     v = K.mean(vv, axis=-1)
#     return v



def get_compiled(loss='categorical_crossentropy', optimizer='adam',
                 metrics=["categorical_accuracy", invasion_acc,
                          invasion_precision, invasion_recall, invasion_fmeasure],
                 weights=None, **kwargs):
    model = get_model(weights=weights, **kwargs)
    model.compile(loss=loss, optimizer=optimizer,
                  metrics=[loss] + metrics)
    return model


if __name__ == '__main__':
    model = get_compiled()
