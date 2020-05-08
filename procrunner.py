import tensorflow as tf
import os
import numpy as np
import time
import math
import gym
from gym.spaces.box import Box
import atari_wrappers
from multiprocessing import Pipe, Process

def worker(env_name, pipe, atari=False):
    if atari:
        env = atari_wrappers.wrap_deepmind(atari_wrappers.make_atari(env_name), episode_life=False, clip_rewards=False, frame_stack=True, scale=True)
    else:
        env = gym.make(env_name)
    s = env.reset()
    reward = 0
    done = False
    try:
        while True:
            pipe.send((s, reward, done))
            cmd, data = pipe.recv()
            if cmd == 'step':
                if isinstance(env.action_space, Box):
                    data = np.clip(data, env.action_space.low, env.action_space.high)
                s, reward, done, _ = env.step(data)
            else:
                break
            if done:
                s = env.reset()
    finally:
        pipe.close()
        env.close()


class ProcRunner:
    def __init__(self, env_name, env_count, update_interval, writer=None, atari=False):
        self.update_interval = update_interval
        self.env_count = env_count
        self.p_pipe, c_pipe = zip(*[Pipe() for _ in range(env_count)])
        self.workers = [Process(target=worker, args=(env_name, c_pipe[i], atari), daemon=True) for i in range(env_count)]
        for w in self.workers:
            w.start()
        self.states = list()
        self.total_reward = [0 for _ in range(env_count)]
        self.writer = writer
        for p in self.p_pipe:
            s, _, _ = p.recv()
            self.states.append(s)

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
        s_lst = [list() for _ in range(self.env_count)]
        a_lst = [list() for _ in range(self.env_count)]
        r_lst = [list() for _ in range(self.env_count)]
        done_lst = [list() for _ in range(self.env_count)]
        v_lst = [list() for _ in range(self.env_count)]
        action_prob_lst = [list() for _ in range(self.env_count)]


        for _ in range(self.update_interval):
            currstep += 1
            action, action_prob, value = model.get_actions(self.states)
            for i in range(self.env_count):
                self.p_pipe[i].send(('step',action[i]))

            for i in range(self.env_count):
                ns, reward, done = self.p_pipe[i].recv()
                self.total_reward[i] += reward
                s_lst[i].append(np.copy(self.states[i]))
                a_lst[i].append(action[i])
                r_lst[i].append(reward)
                v_lst[i].append(value[i])
                action_prob_lst[i].append(action_prob[i])
                done_lst[i].append(0 if done else 1)
                self.states[i] = ns
                if done:
                    if self.writer != None:
                        score_summary_data = tf.Summary(value=[tf.Summary.Value(tag="score", simple_value=self.total_reward[i])])
                        self.writer.add_summary(score_summary_data, currstep)
                    self.avg += self.total_reward[i]
                    self.cnt += 1
                    if self.total_reward[i] > self.high:
                        self.high = self.total_reward[i]
                    self.total_reward[i] = 0
                    i += 1
        last_values = model.get_value(self.states)
        for i in range(self.env_count):
            v_lst[i].append(last_values[i])

        return [[s_lst[i], a_lst[i], r_lst[i], done_lst[i], v_lst[i], action_prob_lst[i]] for i in range(self.env_count)] 

    def close(self):
        for p in self.p_pipe:
            p.send(('exit', 0))
            p.close()