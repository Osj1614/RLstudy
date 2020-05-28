import gym
import pybullet_envs
from gym.spaces.box import Box
import tensorflow as tf
from rnd import RND
from trainer import train, run_only
import actiontype
import models
from running_std import RunningMeanStd

def main():
    env_name = 'HumanoidBulletEnv-v0'
    env = gym.make(env_name)
    if isinstance(env.action_space, Box):
        output_size = env.action_space.shape[0]
    else:
        output_size = env.action_space.n

    with tf.Session() as sess:
        name = 'humanoid_rnd'
        with tf.variable_scope(name):
            input = tf.placeholder(tf.float32, [None, env.observation_space.shape[0]])
            network = models.mlp(input)

            state_rms = RunningMeanStd(sess, shape=env.observation_space.shape[0])
            norm_input = tf.clip_by_value((input - state_rms._mean) / tf.sqrt(state_rms._var), -10, 10)
            with tf.variable_scope('target'):
                target_net = models.add_dense(models.mlp(norm_input), 256, name='dense2')
            with tf.variable_scope('predict'):
                predict_net = models.add_dense(models.mlp(norm_input), 256, name='dense2')
            

            #model = RND(sess, input, state_rms, network, actiontype.Discrete, output_size, target_net, predict_net, epochs=4, minibatch_size=8, gamma=0.99, beta2=0.01, epsilon=0.1,\
            #    learning_rate=lambda f : 2.5e-4 * (1-f), name=name)
            model = RND(sess, input, state_rms, network, actiontype.Continuous, output_size, target_net, predict_net, epochs=10, minibatch_size=32, gamma=0.99, beta2=0.000, epsilon=0.2, \
                 learning_rate=lambda f : 3e-4*(1-f), name=name)
        train(sess, model, env_name, 50000000, 2048, num_envs=16, log_interval=5)   
        #train(sess, model, env_name, 1000000, 128, num_envs=4)
        #run_only(sess, model, env, render=True)
        env.close()

if __name__ == "__main__":
    main()