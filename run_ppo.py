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
    env_name = 'CartPole-v1'
    env = gym.make(env_name)
    output_size = env.action_space.n

    with tf.Session() as sess:
        name = 'cpr'
        with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
            nenvs = 16
            minibatches = 8
            nsteps = 128
            def policy_fn(obs, nenvs):
                return models.lstm(nenvs, 64)(models.mlp()(obs))
                #return models.mlp()(obs)
            network = models.ACnet(env.observation_space.shape, policy_fn, nenvs, nsteps, minibatches, actiontype.Discrete, output_size, recurrent=True)

            model = PPO(sess, network, epochs=4, gamma=0.99, beta2=0.01, epsilon=0.1,\
                learning_rate=lambda f : 2.5e-4 * (1-f), name=name)
            
        train(sess, model, env_name, 1e6)
        #run_only(sess, model, env, render=True)
        env.close()

if __name__ == "__main__":
    main()