import gym
import sys
import pybullet_envs
import tensorflow as tf
import numpy as np
from rl.ppo import PPO
from rl.rnd import RND
from rl.trainer import train, run_only
from rl.atari_wrappers import wrap_deepmind, make_atari
from rl import actiontype
from rl import models
from rl.running_std import RunningMeanStd

def main():
    env_name = 'BreakoutNoFrameskip-v4'
    env = wrap_deepmind(make_atari(env_name))
    output_size = env.action_space.n

    name = 'breakout2'
    with tf.Session() as sess:
        with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
            nenvs = 16
            minibatches = 4
            nsteps = 128
            def policy_fn(obs, nenvs):
                #return models.nature_cnn()(obs)
                return models.atari_lstm(nenvs, 512)(obs)
                #return models.mlp()(obs)
            network = models.ACnet(env.observation_space.shape, policy_fn, nenvs, nsteps, minibatches, actiontype.Discrete, output_size, recurrent=True)

            model = PPO(sess, network, epochs=4, epsilon=0.1,\
                learning_rate=lambda f : 2.5e-4 * (1-f), name=name)
            
        train(sess, model, env_name, 10e6, atari=True)
        #run_only(sess, model, env, render=True)
        env.close()

if __name__ == "__main__":
    main()