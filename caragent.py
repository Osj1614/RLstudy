import numpy as np
import tensorflow as tf
import gym
import threading
from continuous_ppo import ContinuousPPO

def playgame(model, env):
    s = env.reset()
    reward_sum = 0
    while True:
        env.render()
        s = [(s[0] + 1.2) / 1.8, s[1] * 100 / 7]
        a, _ = model.getAction(s)
        s, reward, done, _ = env.step(a)
        reward_sum += reward
        if done:
            print("Total score: {}".format(reward_sum))
            break

def train(model, env):
    rewards = 0
    for i in range(num_episodes):
        s = env.reset()
        s = [(s[0] + 1.2) / 1.8, s[1] * 100 / 7]
        done = False

        while not done:
            s_lst = []
            a_lst = []
            r_lst = []
            done_lst = []
            oldpi = []
            max_v = 0

            for _ in range(update_interval):
                if abs(s[1]) > max_v:
                    max_v = abs(s[1])
                action, action_prob = model.getAction(s)
                ns, reward, done, _ = env.step(action)
                s_lst.append(s)
                a_lst.append(action)
                reward = reward / 100
                r_lst.append(reward)
                oldpi.append(action_prob)
                done_lst.append(0 if done else 1)
                rewards += reward * 100
                s = ns
                s = [(s[0] + 1.2) / 1.8, s[1] * 100 / 7]
                if done:
                    break
            s_lst.append(s)
            model.TrainBatch(s_lst, a_lst, r_lst, done_lst, oldpi, 3)
        
        if i % 20 == 19:
            print("Episode: {} reward: {}".format(i, rewards / 20))
            if rewards / 20 > 95:
                rewards = 0
                break
    env.close()

environment = gym.make('MountainCarContinuous-v0')
input_size = environment.observation_space.shape[0]
output_size = 1
num_episodes = 1000
update_interval = 20

input = tf.placeholder(tf.float32, [None, input_size])
network = tf.layers.dense(input, 128, activation=tf.nn.relu)

with tf.Session() as sess:
    model = ContinuousPPO(sess, input, network, output_size)
    sess.run(tf.global_variables_initializer())
    train(model, environment)
    playgame(model, environment)
environment.close()