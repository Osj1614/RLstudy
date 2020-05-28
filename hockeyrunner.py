import numpy as np
import time
import airhockey
import math
from multiprocessing import Process, Pipe

fps = 20

def worker(pipe):
    env = airhockey.AirHockey()
    
    s, s2 = env.reset()
    r = 0
    r2 = 0
    done = False
    try:
        while True:
            pipe.send((s, s2, r, r2, done))
            cmd, data, data2 = pipe.recv()
            if cmd == 'step':
                data = np.clip(data, env.action_space.low, env.action_space.high)
                data2 = np.clip(data2, env.action_space.low, env.action_space.high)
                s, s2, r, r2, done = env.step(data, data2, 1/fps)
            else:
                break
            if done:
                s, s2 = env.reset()
    finally:
        pipe.close()
        env.close()

def play_game(model, model2):
    env = airhockey.AirHockey(render=True)
    state, state2 = env.reset()
    done = False
    while not done:
        action = model.get_action(state)
        action2 = model2.get_action(state2)
        for _ in range(int(60/fps)):
            env.render()
            ns, ns2, _, _, done = env.step(np.clip(action, env.action_space.low, env.action_space.high), \
                np.clip(action2, env.action_space.low, env.action_space.high), (1/60))
        state = ns
        state2 = ns2

def play_human(model):
    env = airhockey.AirHockey(render=True, human=True)
    _, state = env.reset()
    done = False
    while not done:
        env.render()
        action = model.get_action(state)
        
        for _ in range(int(60/fps)):
            env.render()
            pos = np.array(env.mouse.get_pos())
            wh = np.array((airhockey.width, airhockey.height/2)) * 10
            _, ns, _, _, done = env.step(np.clip((pos-wh-(0, airhockey.height*10))/wh, -1, 1), \
                np.clip(action, env.action_space.low, env.action_space.high) ,(1/60))
        state = ns

class Runner:
    def __init__(self, env_count, update_interval):
        self.update_interval = update_interval
        self.env_count = env_count
        self.p_pipe, c_pipe = zip(*[Pipe() for _ in range(env_count)])
        self.workers = [Process(target=worker, args=(c_pipe[i],), daemon=True) for i in range(env_count)]
        for w in self.workers:
            w.start()
        self.states = list()
        self.states2 = list()
        for p in self.p_pipe:
            s, s2, _, _, _ = p.recv()
            self.states.append(s)
            self.states2.append(s2)

    def run_steps(self, model, model2, currstep=0):
        s_lst = [list() for _ in range(self.env_count)]
        a_lst = [list() for _ in range(self.env_count)]
        r_lst = [list() for _ in range(self.env_count)]
        done_lst = [list() for _ in range(self.env_count)]
        v_lst = [list() for _ in range(self.env_count)]
        action_prob_lst = [list() for _ in range(self.env_count)]

        s_lst2 = [list() for _ in range(self.env_count)]
        a_lst2 = [list() for _ in range(self.env_count)]
        r_lst2 = [list() for _ in range(self.env_count)]
        done_lst2 = [list() for _ in range(self.env_count)]
        v_lst2 = [list() for _ in range(self.env_count)]
        action_prob_lst2 = [list() for _ in range(self.env_count)]


        for _ in range(self.update_interval):
            currstep += 1
            action, action_prob, value = model.get_actions(self.states)
            action2, action_prob2, value2 = model2.get_actions(self.states2)
            for i in range(self.env_count):
                self.p_pipe[i].send(('step',action[i],action2[i]))

            for i in range(self.env_count):
                ns, ns2, r, r2, done = self.p_pipe[i].recv()

                s_lst[i].append(np.copy(self.states[i]))
                a_lst[i].append(action[i])
                r_lst[i].append(r)
                v_lst[i].append(value[i])
                action_prob_lst[i].append(action_prob[i])
                done_lst[i].append(0 if done else 1)
                self.states[i] = ns
                
                s_lst2[i].append(np.copy(self.states2[i]))
                a_lst2[i].append(action2[i])
                r_lst2[i].append(r2)
                v_lst2[i].append(value2[i])
                action_prob_lst2[i].append(action_prob2[i])
                done_lst2[i].append(0 if done else 1)
                self.states2[i] = ns2

        last_values = model.get_value(self.states)
        last_values2 = model2.get_value(self.states2)
        for i in range(self.env_count):
            v_lst[i].append(last_values[i])
            v_lst2[i].append(last_values2[i])

        return [[s_lst[i], a_lst[i], r_lst[i], done_lst[i], v_lst[i], action_prob_lst[i]] for i in range(self.env_count)],\
             [[s_lst2[i], a_lst2[i], r_lst2[i], done_lst2[i], v_lst2[i], action_prob_lst2[i]] for i in range(self.env_count)]

    def close(self):
        for p in self.p_pipe:
            p.send(('exit', 0))
            p.close()