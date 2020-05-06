import numpy as np
import tensorflow as tf
import gym
import threading
import time
import sys
from a2c import ActorCritic
from ppo import PPO

ATARI = True

def add_dense(inputs, output_size, activation=None, kernel_initializer=None, name=""):
    layer = tf.layers.dense(inputs, output_size, activation=activation, kernel_initializer=kernel_initializer, name=name)
    with tf.variable_scope(name, reuse=True):
        tf.summary.histogram("kernel", tf.get_variable("kernel"))
        tf.summary.histogram("bias", tf.get_variable("bias"))
    return layer

def add_cnn(inputs, filters, kernel_size, strides=(1,1), padding='same', activation=None, kernel_initializer=None, name=""):
    layer = tf.layers.conv2d(inputs, filters, kernel_size, strides=strides, padding=padding, activation=activation, kernel_initializer=kernel_initializer, name=name)
    with tf.variable_scope(name, reuse=True):
        tf.summary.histogram("kernel", tf.get_variable("kernel"))
        tf.summary.histogram("bias", tf.get_variable("bias"))
    return layer

def atari_ram(state):
    if not ATARI:
        return state
    size = state.shape[0]
    new_state = np.zeros(size * 8)
    for i in range(0, 8):
        new_state[i*size:(i+1)*size] = state % 2
        state = state // 2
    return new_state

def playgame(model, env):
    s = env.reset()
    reward_sum = 0
    while True:
        env.render()
        time.sleep(0.001)
        a, _, v = model.get_action(atari_ram(s))
        s, reward, done, _ = env.step(a)
        reward_sum += reward
        if done:
            print("Total score: {}".format(reward_sum))
            break

def train(sess, model, env, num_episodes, update_interval):
    model.make_summary()
    summaries = tf.summary.merge_all()
    writer = tf.summary.FileWriter("./logs/" + model.name, sess.graph)
    total_train = 0
    s_lst = list()
    a_lst = list()
    r_lst = list()
    done_lst = list()
    v_lst = list()
    action_prob_lst = list()
    i = 0
    s = env.reset()
    if ATARI:
        s = atari_ram(s)
    total_reward = 0
    while i < num_episodes:
        s_lst.clear()
        a_lst.clear()
        r_lst.clear()
        done_lst.clear()
        v_lst.clear()
        action_prob_lst.clear()
        for _ in range(update_interval):
            action, action_prob, value = model.get_action(s)
            ns, reward, done, _ = env.step(action)
            s_lst.append(s)
            a_lst.append(action)
            v_lst.append(value)
            r_lst.append(reward)
            action_prob_lst.append(action_prob)
            done_lst.append(0.0 if done else 1.0)
            s = atari_ram(ns)
            total_reward += reward
            total_train += 1
            if done:
                print(f"Episode {i}: {total_reward}")
                score_summary_data = tf.Summary(value=[tf.Summary.Value(tag="score", simple_value=total_reward)])
                writer.add_summary(score_summary_data, i)
                if i % 50 == 0:
                    playgame(model, env)
                i += 1

                s = atari_ram(env.reset())
                total_reward = 0
        v_lst.append(np.squeeze(model.get_value([s])))
        summary_data = model.train_batch(s_lst, a_lst, r_lst, done_lst, v_lst, action_prob_lst, summaries)
        writer.add_summary(summary_data, total_train)

def main():
    environment = gym.make('Pong-ram-v0')
    input_size = environment.observation_space.shape[0]
    if ATARI:
        input_size = input_size * 8
    output_size = environment.action_space.n

    with tf.Session() as sess:
        with tf.variable_scope('Pongram'):
            input = tf.placeholder(tf.float32, [None, input_size])
            network = add_dense(input, 64, activation=tf.nn.tanh, kernel_initializer=tf.orthogonal_initializer(np.sqrt(2)), name="dense1")
            network = add_dense(network, 64, activation=tf.nn.tanh, kernel_initializer=tf.orthogonal_initializer(np.sqrt(2)), name="dense2")
            model = PPO(sess, input, network, 2, output_size, epochs=4, minibatch_size=4, gamma=0.99, beta2=0.01, name="Pongram")
        sess.run(tf.global_variables_initializer())
        train(sess, model, environment, 4000000, 128)
        playgame(model, environment)
    environment.close()

main()