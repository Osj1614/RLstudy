import tensorflow as tf
import numpy as np

LOG_WEIGHT = False

def add_dense(inputs, output_size, activation=None, kernel_initializer=tf.orthogonal_initializer(np.sqrt(2)), name=""):
    layer = tf.layers.dense(inputs, output_size, activation=activation, kernel_initializer=kernel_initializer, name=name)
    if LOG_WEIGHT:
        with tf.variable_scope(name, reuse=True):
            tf.summary.histogram("kernel", tf.get_variable("kernel"))
            tf.summary.histogram("bias", tf.get_variable("bias"))
    return layer

def add_cnn(inputs, filters, kernel_size, strides=(1,1), padding='valid', data_format='channels_last', activation=tf.orthogonal_initializer(np.sqrt(2)), kernel_initializer=None, name=""):
    layer = tf.layers.conv2d(inputs, filters, kernel_size, strides=strides, padding=padding, data_format=data_format, activation=activation, kernel_initializer=kernel_initializer, name=name)
    if LOG_WEIGHT:
        with tf.variable_scope(name, reuse=True):
            tf.summary.histogram("kernel", tf.get_variable("kernel"))
            tf.summary.histogram("bias", tf.get_variable("bias"))
    return layer

def mlp(input, nodes=64, depth=2):
    network = add_dense(input, nodes, activation=tf.nn.tanh, name=f'dense0')
    for i in range(1, depth):
        network = add_dense(network, nodes, activation=tf.nn.tanh, name=f'dense{i}')
    return network

def nature_cnn(input, name='cnn'):
    network = add_cnn(input, 32, (8, 8), strides=(4, 4), activation=tf.nn.relu, name=f'cnn0')
    network = add_cnn(network, 64, (4, 4), strides=(2, 2), activation=tf.nn.relu, name=f'cnn1')
    network = add_cnn(network, 64, (3, 3), strides=(1, 1), activation=tf.nn.relu, name=f'cnn2')
    network = tf.contrib.layers.flatten(network)            
    network = add_dense(network, 512, activation=tf.nn.relu, name=f'dense0')
    return network