import tensorflow as tf
import os
import numpy as np
from gym.spaces import Box
import time
import math

class Runner:
    def __init__(self, env, update_interval):
        self.env = env
        self.update_interval = update_interval
        self.state = env.reset()
        self.total_reward = 0
        self.clip = isinstance(env.action_space, Box)
        self.avg = 0
        self.high = -1000000
        self.cnt = 0
        
    def get_avg_high(self):
        avg = self.avg / (self.cnt+1e-8)
        high = self.high
        self.avg = 0
        self.high = -1000000
        self.cnt = 0
        return avg, high

    def run_steps(self, model, currstep=0):
        s_lst = list()
        a_lst = list()
        r_lst = list()
        done_lst = list()
        action_prob_lst = list()
        i = 0
        s = np.reshape(self.state, [-1])
        for _ in range(self.update_interval):
            currstep += 1
            action, action_prob = model.get_action(s)
            if self.clip:
                ns, reward, done, _ = self.env.step(np.clip(action, self.env.action_space.low, self.env.action_space.high))
            else:
                ns, reward, done, _ = self.env.step(action)
            self.total_reward += reward
            s_lst.append(s.copy())
            a_lst.append(action)
            r_lst.append(reward)
            action_prob_lst.append(action_prob)
            done_lst.append(0 if done else 1)
            self.state = ns
            s = np.reshape(self.state, [-1])
            if done:
                self.state = self.env.reset()
                s = np.reshape(self.state, [-1])
                
                self.avg += self.total_reward
                self.cnt += 1
                if self.total_reward > self.high:
                    self.high = self.total_reward
                self.total_reward = 0
                i += 1
        s_lst.append(self.state)
        return [[s_lst, a_lst, r_lst, done_lst, action_prob_lst]]
            
    def playgame(self, model, render=True):
        s = self.env.reset()
        reward_sum = 0
        while True:
            if render:
                self.env.render()
            action = model.get_action(np.reshape(s, [-1]))
            if self.clip:
                ns, reward, done, _ = self.env.step(np.clip(action, self.env.action_space.low, self.env.action_space.high))
            else:
                ns, reward, done, _ = self.env.step(action)
            s = ns
            reward_sum += reward
            if done:
                break
            time.sleep(0.008)
        self.state = self.env.reset()
        self.total_reward = 0
        return reward_sum