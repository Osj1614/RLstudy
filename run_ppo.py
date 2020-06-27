import gym
import sys
import pybullet_envs
import tensorflow as tf
import numpy as np
from rl.ppo import PPO
from rl.rnd import RND
from rl.trainer import train, run_only
from rl import actiontype
from rl import models
from rl.running_std import RunningMeanStd

def main():
    env_name = 'LunarLander-v2'
    env = gym.make(env_name)
    output_size = env.action_space.n

    name = 'lunarr'
    with tf.Session() as sess:
        with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
            nenvs = 16
            minibatches = 8
            nsteps = 128
            def policy_fn(obs, nenvs):
                return models.lstm(nenvs, 64)(models.mlp()(obs))
                #return models.mlp()(obs)
            network = models.ACnet(env.observation_space.shape, policy_fn, nenvs, nsteps, minibatches, actiontype.Discrete, output_size, recurrent=True)

            model = PPO(sess, network, epochs=8, epsilon=0.1,\
                learning_rate=lambda f : 2.5e-4 * (1-f), name=name)
            
        train(sess, model, env_name, 5e6)
        #run_only(sess, model, env, render=True)
        env.close()

if __name__ == "__main__":
    main()