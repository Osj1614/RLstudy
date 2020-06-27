import gym
import pybullet_envs
from gym.spaces.box import Box
import tensorflow as tf
from rnd import RND
from trainer import train, run_only
import atari_wrappers
import actiontype
import models
from running_std import RunningMeanStd

def main():
    run = True
    state = 2
    env_name = 'HumanoidFlagrunBulletEnv-v0'
    if state == 0:
        env = atari_wrappers.wrap_deepmind(atari_wrappers.make_atari(env_name), episode_life=True, clip_rewards=True, frame_stack=True, scale=True)
    else:    
        env = gym.make(env_name)
    if isinstance(env.action_space, Box):
        output_size = env.action_space.shape[0]
    else:
        output_size = env.action_space.n

    with tf.Session() as sess:
        name = 'flag_rnd3'
        with tf.variable_scope(name):
            input = tf.placeholder(tf.float32, [None, *env.observation_space.shape])
            state_rms = RunningMeanStd(sess, shape=env.observation_space.shape)
            norm_input = tf.clip_by_value((input - state_rms._mean) / tf.sqrt(state_rms._var), -5, 5)

            if state == 0:
                with tf.variable_scope('policy'):
                    network = models.nature_cnn(input)
                norm_input = norm_input[:,:,:,0]
                with tf.variable_scope('target'):
                    target_net = models.add_dense(models.nature_cnn(norm_input), 256, name='dense1')
                with tf.variable_scope('predict'):
                    predict_net = models.add_dense(models.nature_cnn(norm_input), 256, name='dense1')
                with tf.variable_scope('value'):
                    value_net = models.nature_cnn(input)
                with tf.variable_scope('value_in'):
                    value_in_net = models.nature_cnn(input)
                model = RND(sess, input, state_rms, network, actiontype.Discrete, output_size, target_net, predict_net, value_in_net,\
                     value_network=value_net, gamma=0.999, learning_rate=lambda f : 0.0001, epochs=4, minibatch_size=4, beta2=0.01, name=name)
            else:
                if state == 1:
                    with tf.variable_scope('policy'):
                        network, seq_len, init_state, last_state = models.lstm(models.mlp(input), 64)
                    with tf.variable_scope('target'):
                        target_net = models.add_dense(models.mlp(norm_input), 256, name='dense2')
                    with tf.variable_scope('predict'):
                        predict_net = models.add_dense(models.mlp(norm_input), 256, name='dense2')
                    with tf.variable_scope('value_in'):
                        value_in_net = models.mlp(input)
                    model = RND(sess, input, state_rms, network, actiontype.Discrete, output_size, target_net, predict_net, value_in_net, epochs=4, minibatch_size=8, gamma=0.99, beta2=0.01, epsilon=0.1,\
                        coef_in=1., learning_rate=lambda f : 2.5e-4*(1-f), name=name, )
                elif state == 2:
                    with tf.variable_scope('policy'):
                        network = models.mlp(norm_input)
                    with tf.variable_scope('target'):
                        target_net = models.add_dense(models.mlp(norm_input), 256, name='dense2')
                    with tf.variable_scope('predict'):
                        predict_net = models.add_dense(models.mlp(norm_input), 256, name='dense2')
                    with tf.variable_scope('value'):
                        value_net = models.mlp(norm_input)
                    with tf.variable_scope('value_in'):
                        value_in_net = models.mlp(norm_input)
                    model = RND(sess, input, state_rms, network, actiontype.Continuous, output_size, target_net, predict_net, value_in_net, value_network=value_net, epochs=10, minibatch_size=32, gamma=0.99, beta2=0.000, epsilon=0.2, \
                        coef_in=.5, learning_rate=lambda f : 3e-4*(1-f), name=name)
        if run:
            run_only(sess, model, env, render=True)
        else:
            if state == 0:
                train(sess, model, env_name, 10000000, 256, num_envs=16, atari=True)
            elif state == 1:
                train(sess, model, env_name, 5e6, 128, num_envs=16)
            elif state == 2:
                train(sess, model, env_name, 100e6, 2048, num_envs=24, log_interval=5)
        env.close()

if __name__ == "__main__":
    main()