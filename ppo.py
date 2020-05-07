import numpy as np
import math
import tensorflow as tf
import actiontype
from running_std import RunningMeanStd

class PPO:
    def __init__(self, sess, state, network, action_type, action_size, value_network=None, name="", learning_rate=0.00025, beta=0.5, beta2=0.01, gamma=0.99, epsilon=0.1, lamda=0.95, max_grad_norm=0.5, epochs=4, minibatch_size=16, use_gae=True, v_clip=True):
        self.state = state
        self.sess = sess
        self.action_type = action_type
        self.action_size = action_size
        self.name = name
        self.learning_rate = learning_rate
        self.beta = beta
        self.beta2 = beta2
        self.gamma = gamma
        self.epsilon = epsilon
        self.lamda = lamda
        self.max_grad_norm = max_grad_norm
        self.epochs = epochs
        self.minibatch_size=  minibatch_size
        self.v_clip = v_clip
        self.cumulative_reward = 0
        self.use_gae = use_gae

        with tf.variable_scope("ppo"):
            if action_type == actiontype.Continuous:
                self.action = actiontype.continuous(network, action_size)
                self.actions = tf.placeholder(tf.float32, [None, self.action_size], name="Action")
            else:
                self.action = actiontype.discrete(network, action_size)
                self.actions = tf.placeholder(tf.int32, [None], name="Action")
                self.action_size = 1
            self.reward_std = RunningMeanStd(sess, scope="cumulative_reward")
            self.sample = self.action.sample()
            self.neglogp = self.action.neglogp(self.sample)
            self.bulid_train(network, value_network)
        
    def bulid_train(self, network, value_network=None):
        self.advantage = tf.placeholder(tf.float32, [None], name="Advantage")
        self.old_value = tf.placeholder(tf.float32, [None], name="Old_value")
        self.returns = tf.placeholder(tf.float32, [None], name="Returns")
        self.prevneglogp = tf.placeholder(tf.float32, [None], name="Old_pi_a")
        self.lr = tf.placeholder(tf.float32, [], name="Learning_rate")

        if value_network == None:
            self.value = tf.layers.dense(network, 1, kernel_initializer=tf.orthogonal_initializer(), name="Value")
        else:
            self.value = tf.layers.dense(value_network, 1, kernel_initializer=tf.orthogonal_initializer(), name="Value")
        self.value = self.value[:, 0]
        with tf.variable_scope('Actor_loss'):
            pi_a = self.action.neglogp(self.actions)
            ratio = tf.exp(self.prevneglogp - pi_a)
            actor_loss = ratio * -self.advantage
            clipped_loss = tf.clip_by_value(ratio, 1 - self.epsilon, 1 + self.epsilon) * -self.advantage
            self.actor_loss = tf.reduce_mean(tf.maximum(actor_loss, clipped_loss))
            self.clipfrac = tf.reduce_mean(tf.to_float(tf.greater(tf.abs(ratio - 1.0), self.epsilon)))

        with tf.variable_scope('Entropy'):
            self.entropy = tf.reduce_mean(self.action.entropy())
        with tf.variable_scope('Critic_loss'):
            critic_loss1 = tf.squared_difference(self.returns, self.value)
            if self.v_clip:
                critic_loss2 = tf.squared_difference(self.returns, self.old_value + tf.clip_by_value(self.value - self.old_value, -self.epsilon, self.epsilon))
                self.vclipfrac = tf.reduce_mean(tf.to_float(tf.greater(tf.abs(self.value - self.old_value), self.epsilon)))
                self.critic_loss = tf.reduce_mean(tf.maximum(critic_loss1, critic_loss2)) * 0.5
            else:
                self.critic_loss = tf.reduce_mean(critic_loss1)
        with tf.variable_scope('Total_loss'):
            self.loss =  self.actor_loss - self.entropy * self.beta2 + self.critic_loss * self.beta
        
        params = tf.trainable_variables(self.name)
        with tf.variable_scope('train'):
            trainer = tf.train.AdamOptimizer(learning_rate=self.lr, epsilon=1e-5)
            grads_and_var = trainer.compute_gradients(self.loss, params)
            grads, var = zip(*grads_and_var)
            if self.max_grad_norm != None:
                grads, _ = tf.clip_by_global_norm(grads, self.max_grad_norm)
            grads_and_var = list(zip(grads, var))
            self.train = trainer.apply_gradients(grads_and_var)

    def make_summary(self):
        tf.summary.scalar('loss', self.loss)
        tf.summary.scalar('actor_loss', self.actor_loss)
        tf.summary.scalar('critic_loss', self.critic_loss)
        tf.summary.scalar('entropy', self.entropy)
        tf.summary.scalar('value', tf.reduce_mean(self.value))
        tf.summary.scalar('clip_fraction', self.clipfrac)
        tf.summary.scalar('learning_rate', self.lr)
        if self.v_clip:
            tf.summary.scalar('value_clip_fraction', self.vclipfrac)
        

    def get_value(self, state):
        feed_dict = {self.state : state}
        return self.sess.run(self.value, feed_dict)

    def get_action(self, state):
        a, ap, v = self.sess.run([self.sample, self.neglogp, self.value], feed_dict={self.state : [state]})
        a = np.squeeze(a, 0)
        ap = np.squeeze(ap, 0)
        v = np.squeeze(v, 0)
        return a, ap, v

    def get_actions(self, states):
        a, ap, v = self.sess.run([self.sample, self.neglogp, self.value], feed_dict={self.state : states})
        return a, ap, v

    def run_trains(self, returns_lst, advantage_lst, action_prob_lst, value_lst, a_lst, s_lst, learning_rate):
        end = 0
        size = len(returns_lst)
        order = np.arange(size)
        for _ in range(self.epochs):
            np.random.shuffle(order)
            for i in range(0, size, self.minibatch_size):
                end = i + self.minibatch_size
                if end <= size:
                    ind = order[i:end]
                    slices = (arr[ind] for arr in (returns_lst, advantage_lst, action_prob_lst, value_lst, a_lst, s_lst))
                    self.run_train(*slices, learning_rate)

    def run_train(self, returns_lst, advantage_lst, action_prob_lst, value_lst, a_lst, s_lst, learning_rate):
        self.sess.run(self.train, feed_dict={self.returns:returns_lst, \
            self.advantage:advantage_lst, self.prevneglogp:action_prob_lst, self.old_value:value_lst, \
            self.actions:a_lst, self.state : s_lst, self.lr : learning_rate})

    def calc_gae(self, r_lst, value_lst, done_lst):
        cur = 0.0
        size = len(r_lst)
        advantage_lst = np.zeros(size)
        for i in reversed(range(size)):
            delta = r_lst[i] + self.gamma * value_lst[i+1] * done_lst[i] - value_lst[i]
            advantage_lst[i] = cur = self.lamda * self.gamma * done_lst[i] * cur + delta
        return advantage_lst



    def train_batch(self, s_lst, a_lst, r_lst, done_lst, value_lst, action_prob_lst, learning_rate=None, summaries=None):
        if learning_rate == None:
            learning_rate = self.learning_rate
        s_lst = np.clip(np.asarray(s_lst, dtype=np.float32), -5, 5)
        r_lst = np.asarray(r_lst, dtype=np.float32)
        a_lst = np.asarray(a_lst, dtype=np.int32)
        value_lst = np.asarray(value_lst, dtype=np.float32)
        done_lst = np.asarray(done_lst, dtype=np.int32)
        action_prob_lst = np.asarray(action_prob_lst, dtype=np.float32)
        size = len(done_lst)
        returns_lst = np.zeros([size])
        advantage_lst = np.zeros([size])
        cumulative_lst = np.zeros([size])
        for i in range(size):
            self.cumulative_reward = r_lst[i] + self.cumulative_reward * self.gamma
            cumulative_lst[i] = self.cumulative_reward
            self.cumulative_reward *= done_lst[i]

        self.reward_std.update(cumulative_lst)
        r_lst /= math.sqrt(self.reward_std.var)
        r_lst = np.clip(r_lst, -5, 5)
        if self.use_gae: #GAE
            advantage_lst = self.calc_gae(r_lst, value_lst, done_lst)
            value_lst = value_lst[:-1]
            returns_lst = value_lst + advantage_lst
            advantage_lst = (advantage_lst - advantage_lst.mean()) / (advantage_lst.std() + 1e-8)
        else:
            returns_lst = r_lst + self.gamma * value_lst[1:] * done_lst
            value_lst = value_lst[:-1]
            advantage_lst = returns_lst - value_lst

        self.run_trains(returns_lst, advantage_lst, action_prob_lst, value_lst, a_lst, s_lst, learning_rate)
        return self.sess.run(summaries, feed_dict={self.returns:returns_lst, self.actions:a_lst, self.advantage:advantage_lst, \
            self.prevneglogp:action_prob_lst, self.old_value:value_lst, self.state : s_lst, self.lr : learning_rate})

    def train_batches(self, batch_lst, learning_rate=None, summaries=None):
        if learning_rate == None:
            learning_rate = self.learning_rate

        if not isinstance(self.cumulative_reward, list) or len(self.cumulative_reward) != len(batch_lst):
            self.cumulative_reward = [0 for _ in range(len(batch_lst))]

        for i in range(len(batch_lst)):
            size = len(batch_lst[i][1])
            cumulative_lst = np.zeros([size])
            for j in range(size):
                self.cumulative_reward[i] = batch_lst[i][2][j] + self.cumulative_reward[i] * self.gamma
                cumulative_lst[j] = self.cumulative_reward[i]
                self.cumulative_reward[i] *= batch_lst[i][3][j]

            self.reward_std.update(cumulative_lst)
        
        s_lsts = np.empty(shape=[0, *np.shape(batch_lst[0][0][0])])
        a_lsts = np.empty(shape=[0, *np.shape(batch_lst[0][1][0])])
        advantage_lsts = np.empty([0])
        action_prob_lsts = np.empty([0])
        value_lsts = np.empty([0])
        returns_lsts = np.empty([0])
        for batch in batch_lst:
            s_lst = np.clip(np.asarray(batch[0], dtype=np.float32), -5, 5)
            if self.action_type == actiontype.Discrete:
                a_lst = np.asarray(batch[1], dtype=np.int32)
            else:
                a_lst = np.asarray(batch[1], dtype=np.float32)
            r_lst = np.asarray(batch[2], dtype=np.float32)
            r_lst /= math.sqrt(self.reward_std.var)
            r_lst =  np.clip(r_lst, -5 ,5)
            done_lst = np.asarray(batch[3], dtype=np.int32)
            value_lst = np.asarray(batch[4], dtype=np.float32)
            action_prob_lst = np.asarray(batch[5], dtype=np.float32)
            size = len(done_lst)
            
            advantage_lst = np.zeros([size])
            if self.use_gae: #GAE
                advantage_lst = self.calc_gae(r_lst, value_lst, done_lst)
                value_lst = value_lst[:-1]
                returns_lst = value_lst + advantage_lst
                advantage_lst = (advantage_lst - advantage_lst.mean()) / (advantage_lst.std() + 1e-8)
            else:
                returns_lst = r_lst + self.gamma * value_lst[1:] * done_lst
                value_lst = value_lst[:-1]
                advantage_lst = returns_lst - value_lst

            s_lsts = np.concatenate((s_lsts, s_lst), axis=0)
            a_lsts = np.concatenate((a_lsts, a_lst), axis=0)
            value_lsts = np.concatenate((value_lsts, value_lst), axis=0)
            action_prob_lsts = np.concatenate((action_prob_lsts, action_prob_lst), axis=0)
            advantage_lsts = np.concatenate((advantage_lsts, advantage_lst), axis=0)
            returns_lsts = np.concatenate((returns_lsts, returns_lst), axis=0)

        self.run_trains(returns_lsts, advantage_lsts, action_prob_lsts, value_lsts, a_lsts, s_lsts, learning_rate)
        return self.sess.run(summaries, feed_dict={self.returns:returns_lsts, self.actions:a_lsts, self.advantage:advantage_lsts, \
            self.prevneglogp:action_prob_lsts, self.old_value:value_lsts, self.state : s_lsts, self.lr : learning_rate})
        
       