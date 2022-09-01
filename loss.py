#import keras
import numpy as np
from keras import losses
import keras.backend as K
import tensorflow as tf


def custom_loss(y_true,y_pred, index=0):
    '''
    index is 0 for european option pricing
    in case of superhedge, there are [0,N] instances you want to consider
    '''
    #return losses.mean_squared_error(y_true[0], y_pred[0])
    z = y_pred[:,index]-y_true[:,index]
    z = K.mean(K.square(z))
    return z


def custom_loss_v2(y_true,y_pred, sorted_node_list):
    '''
    index is 0 for european option pricing
    in case of superhedge, there are [0,N] instances you want to consider
    '''
    #return losses.mean_squared_error(y_true[0], y_pred[0])

    #y_pred = np.array([y_pred[i][x] for i,x in enumerate(sorted_node_list)])
    #y_true = np.array([y_true[i][x] for i,x in enumerate(sorted_node_list)])

    y_pred = tf.gather_nd(y_pred, sorted_node_list, batch_dims=0, name=None)
    y_true = tf.gather_nd(y_true, sorted_node_list, batch_dims=0, name=None)

    z = y_pred - y_true
    z = K.mean(K.square(z))

    return z