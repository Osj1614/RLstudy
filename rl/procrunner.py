import numpy as np
import gym
from gym.spaces.box import Box
from .atari_wrappers import make_atari, wrap_deepmind
from multiprocessing import Pipe, Process

def worker(env_name, pipe, atari=False):
    if atari:
        env = wrap_deepmind(make_atari(env_name))
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
    def __init__(self, model, env_name, atari=False):
        self.update_interval = model.network.nsteps
        self.model = model
        self.nenvs = model.network.nenvs
        self.p_pipe, c_pipe = zip(*[Pipe() for _ in range(self.nenvs)])
        self.workers = [Process(target=worker, args=(env_name, c_pipe[i], atari), daemon=True) for i in range(self.nenvs)]
        for w in self.workers:
            w.start()
        self.states = list()
        self.pdone = list()
        self.total_reward = [0 for _ in range(self.nenvs)]
        for p in self.p_pipe:
            s, _, _ = p.recv()
            self.states.append(s)
            self.pdone.append(0)
        
        if self.model.network.recurrent:
            self.hs = self.model.network.initial_state
                
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
        

    def run_steps(self, currstep=0):
        s_lst = [list() for _ in range(self.nenvs)]
        a_lst = [list() for _ in range(self.nenvs)]
        r_lst = [list() for _ in range(self.nenvs)]
        done_lst = [list() for _ in range(self.nenvs)]
        v_lst = [list() for _ in range(self.nenvs)]
        action_prob_lst = [list() for _ in range(self.nenvs)]
        if self.model.network.recurrent:
            prev_hs = np.copy(self.hs)
        for _ in range(self.update_interval):
            currstep += 1
            if self.model.network.recurrent:
                action, action_prob, values, self.hs = self.model.get_actions(self.states, self.hs, self.pdone)
            else:
                action, action_prob, values = self.model.get_actions(self.states)
            for i in range(self.nenvs):
                self.p_pipe[i].send(('step',action[i]))

            for i in range(self.nenvs):
                ns, reward, done = self.p_pipe[i].recv()
                self.total_reward[i] += reward
                s_lst[i].append(np.copy(self.states[i]))
                a_lst[i].append(action[i])
                r_lst[i].append(reward)
                action_prob_lst[i].append(action_prob[i])
                done_lst[i].append(self.pdone[i])
                v_lst[i].append(values[i])
                self.pdone[i] = 1 if done else 0
                self.states[i] = ns
                if done:
                    self.avg += self.total_reward[i]
                    self.cnt += 1
                    if self.total_reward[i] > self.high:
                        self.high = self.total_reward[i]
                    self.total_reward[i] = 0
                    i += 1

        if self.model.network.recurrent:
            values = self.model.get_value(self.states, self.hs, self.pdone)
        else:
            values = self.model.get_value(self.states)
        for i in range(self.nenvs):
            v_lst[i].append(values[i])
            done_lst[i].append(self.pdone[i])
        batches = [[s_lst[i], a_lst[i], r_lst[i], done_lst[i], action_prob_lst[i], v_lst[i]] for i in range(self.nenvs)]
        if self.model.network.recurrent:
            return batches, prev_hs
        else:
            return batches

    def close(self):
        for p in self.p_pipe:
            p.send(('exit', 0))
            p.close()