import tensorflow as tf
import numpy as np
import gym

env = gym.make("SpaceInvaders-v0")
env.reset()
for i in range(1000):
    env.step(env.action_space.sample())
    env.render('human')
env.close()