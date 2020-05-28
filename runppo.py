import gym
import sys
import pybullet_envs
import tensorflow as tf
import numpy as np
from ppo import PPO
from rnd import RND
from trainer import train, run_only
import actiontype
import models
from running_std import RunningMeanStd

def main():
    env_name = 'LunarLander-v2'
    env = gym.make(env_name)
    output_size = env.action_space.n

    with tf.Session() as sess:
        name = 'lunar'
        with tf.variable_scope(name):
            input = tf.placeholder(tf.float32, [None, env.observation_space.shape[0]])
            network = models.mlp(input)
            model = PPO(sess, input, network, actiontype.Discrete, output_size, epochs=4, minibatch_size=8, gamma=0.99, beta2=0.01, epsilon=0.1,\
                learning_rate=lambda f : 2.5e-4 * (1-f), name=name)
            
        train(sess, model, env_name, 200000, 128, num_envs=16)
        #run_only(sess, model, env, render=True)
        env.close()

if __name__ == "__main__":
    main()