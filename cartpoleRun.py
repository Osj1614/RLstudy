import numpy as np
import tensorflow as tf
import gym
import threading
from a2c import ActorCritic
from ppo import PPO

def playgame(model):
    s = environment.reset()
    reward_sum = 0
    while True:
        environment.render()
        a, _ = model.getAction(s)
        s, reward, done, _ = environment.step(a)
        reward_sum += reward
        if done:
            print("Total score: {}".format(reward_sum))
            break

def train(model):
    sum = 0
    env = gym.make('CartPole-v1')
    for i in range(num_episodes):
        step_count = 0
        s = env.reset()
        done = False

        while not done:
            s_lst = []
            a_lst = []
            r_lst = []
            done_lst = []
            oldpi = []

            for _ in range(update_interval):
                action, action_prob = model.getAction(s)
                ns, reward, done, _ = env.step(action)
                s_lst.append(s)
                a_lst.append(action)
                r_lst.append(reward / 500.0)
                oldpi.append(action_prob)
                done_lst.append(0 if done else 1)
                step_count += 1
                s = ns
                if done:
                    break
            s_lst.append(ns)
            model.TrainBatch(s_lst, a_lst, r_lst, done_lst, oldpi, 3)
        sum += step_count
        if i % 20 == 19:
            print("Episode: {} Step: {}".format(i, sum / 20))
            if sum / 20 > 490:
                break
            sum = 0
    env.close()

environment = gym.make('CartPole-v1')
input_size = environment.observation_space.shape[0]
output_size = environment.action_space.n
num_episodes = 1000
update_interval = 20

input = tf.placeholder(tf.float32, [None, input_size])
network = tf.layers.dense(input, 4, activation=tf.nn.relu)

with tf.Session() as sess:
    model = PPO(sess, input, network, output_size)
    sess.run(tf.global_variables_initializer())
    train(model)
    playgame(model)
environment.close()