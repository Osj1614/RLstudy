import gym
import pybullet_envs
import tensorflow as tf
import numpy as np
from ppo import PPO
from trainer import train, run_only
import actiontype

LOG_WEIGHT = False
DOTEST = True

def add_dense(inputs, output_size, activation=None, kernel_initializer=None, name=""):
    layer = tf.layers.dense(inputs, output_size, activation=activation, kernel_initializer=kernel_initializer, name=name)
    if LOG_WEIGHT:
        with tf.variable_scope(name, reuse=True):
            tf.summary.histogram("kernel", tf.get_variable("kernel"))
            tf.summary.histogram("bias", tf.get_variable("bias"))
    return layer



def main():
    env = gym.make('HumanoidBulletEnv-v0')
    output_size = env.action_space.shape[0]

    with tf.Session() as sess:
        name = 'humanoid_all_mp_5m'
        with tf.variable_scope(name):
            input = tf.placeholder(tf.float32, [None, env.observation_space.shape[0]])
            #initializer = tf.contrib.layers.variance_scaling_initializer(dtype=tf.float32)
            initializer = tf.orthogonal_initializer(np.sqrt(2)) #Orthogonal initializer
            network = add_dense(input, 64, activation=tf.nn.tanh, kernel_initializer=initializer, name="dense1")
            network = add_dense(network, 64, activation=tf.nn.tanh, kernel_initializer=initializer, name="dense2")

            model = PPO(sess, input, network, actiontype.Continuous, output_size, epochs=10, minibatch_size=32, gamma=0.99, beta2=0.000, epsilon=0.2, learning_rate=3e-4, name=name)
        train(sess, model, 'HumanoidBulletEnv-v0', 5000000, 2048, num_envs=4)
        #run_only(sess, model, env)
        env.close()

if __name__ == "__main__":
    main()