import numpy as np
import tensorflow as tf
import gym
from ppo import PPO
import atari_wrappers
import actiontype
from roboschool import train

LOG_WEIGHT = False
DOTEST = True

def add_dense(inputs, output_size, activation=None, kernel_initializer=None, name=""):
    layer = tf.layers.dense(inputs, output_size, activation=activation, kernel_initializer=kernel_initializer, name=name)
    if LOG_WEIGHT:
        with tf.variable_scope(name, reuse=True):
            tf.summary.histogram("kernel", tf.get_variable("kernel"))
            tf.summary.histogram("bias", tf.get_variable("bias"))
    return layer

def add_cnn(inputs, filters, kernel_size, strides=(1,1), padding='valid', data_format='channels_last', activation=None, kernel_initializer=None, name=""):
    layer = tf.layers.conv2d(inputs, filters, kernel_size, strides=strides, padding=padding, data_format=data_format, activation=activation, kernel_initializer=kernel_initializer, name=name)
    if LOG_WEIGHT:
        with tf.variable_scope(name, reuse=True):
            tf.summary.histogram("kernel", tf.get_variable("kernel"))
            tf.summary.histogram("bias", tf.get_variable("bias"))
    return layer

def main():
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    session = tf.Session(config=config)
    input_shape = [84, 84, 4]
    env = atari_wrappers.wrap_deepmind(atari_wrappers.make_atari('BreakoutNoFrameskip-v4'), frame_stack=True, scale=True)
    output_size = env.action_space.n

    with tf.Session() as sess:
        with tf.variable_scope('Breakout_lr'):
            input = tf.placeholder(tf.float32, [None, 84*84*4])
            rinput = tf.reshape(input, [-1, *input_shape])
            
            #initializer = tf.contrib.layers.variance_scaling_initializer(dtype=tf.float32)
            initializer = tf.orthogonal_initializer(np.sqrt(2)) #Orthogonal initializer

            network = add_cnn(rinput, 32, (8, 8), (4, 4), activation=tf.nn.relu, kernel_initializer=initializer, name="cnn1")
            network = add_cnn(network, 64, (4, 4), (2, 2), activation=tf.nn.relu, kernel_initializer=initializer, name="cnn2")
            network = add_cnn(network, 64, (3, 3), (1, 1), activation=tf.nn.relu, kernel_initializer=initializer, name="cnn3")
            network = tf.contrib.layers.flatten(network)            
            network = add_dense(network, 512, activation=tf.nn.relu, kernel_initializer=initializer, name="dense1")

            model = PPO(sess, input, network, actiontype.Discrete, output_size, value_network=None, epochs=4, minibatch_size=4, gamma=0.99, beta2=0.01, name='Breakout_lr')
        sess.run(tf.global_variables_initializer())
        #tf.summary.image('state', rinput[:,:,:,:-1], max_outputs=64)
        train(sess, model, 'BreakoutNoFrameskip-v4', 10000000, 128, num_envs=8, atari=True)

if __name__ == "__main__":
    main()