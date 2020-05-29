import numpy as np
import tensorflow as tf
import gym
from ppo import PPO
import atari_wrappers
import actiontype
from trainer import train, run_only
import models

def main():
    env_name = 'BreakoutNoFrameskip-v4'
    env = atari_wrappers.wrap_deepmind(atari_wrappers.make_atari(env_name), episode_life=False, clip_rewards=False, frame_stack=True, scale=True)
    output_size = env.action_space.n
    input_shape = env.observation_space.shape

    with tf.Session() as sess:
        with tf.variable_scope('Breakout_lr'):
            input = tf.placeholder(tf.float32, [None, *input_shape])

            model = PPO(sess, input, models.nature_cnn(input), actiontype.Discrete, output_size, learning_rate=lambda f : 2.5e-4*(1-f), epochs=4, minibatch_size=4, gamma=0.99, beta2=0.01, name='Breakout_lr')
        train(sess, model, env_name, 1e7, 256, log_interval=5, num_envs=16, atari=True)
        #run_only(sess, model, env, render=True)
        env.close()
        

if __name__ == "__main__":
    main()