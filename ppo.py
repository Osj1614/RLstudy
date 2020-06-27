import numpy as np
import math
import tensorflow as tf
import actiontype
from running_std import RunningMeanStd

class PPO:
    def __init__(self, sess, network, name="", learning_rate=0.00025, beta=0.5, ent_coef=0.01,\
         gamma=0.99, epsilon=0.1, lamda=0.95, max_grad_norm=0.5, epochs=4):
        self.network = network
        self.obs = network.train_obs
        self.sess = sess
        self.name = name
        self.learning_rate = learning_rate
        self.beta = beta
        self.ent_coef = ent_coef
        self.gamma = gamma
        self.epsilon = epsilon
        self.lamda = lamda
        self.max_grad_norm = max_grad_norm
        self.epochs = epochs
        self.cumulative_reward = [0 for _ in range(network.nenvs)]

        with tf.variable_scope("ppo"):
            if network.action.type == actiontype.Continuous:
                self.actions = tf.placeholder(tf.float32, [None, network.action.size], name="Action")
                self.action_size = network.action.size
            else:
                self.actions = tf.placeholder(tf.int32, [None], name="Action")
                self.action_size = 1
            self.global_step = tf.Variable(initial_value=0, dtype=tf.int32, trainable=False, name="global_step")
            self.step_size = tf.placeholder(tf.int32)
            self.increment_global_step = tf.assign_add(self.global_step, self.step_size, name = 'increment_global_step')
            self.reward_rms = RunningMeanStd(sess, scope="cumulative_reward")
            self.value = network.value
            self.sample = network.act_action.sample()
            self.neglogp = network.act_action.neglogp(self.sample)
            self.bulid_train()
            self.make_summary()
        self.writer = tf.summary.FileWriter(f"./logs/{name}", sess.graph)
        
    def bulid_train(self):
        self.advantage = tf.placeholder(tf.float32, [None], name="Advantage")
        self.old_value = tf.placeholder(tf.float32, [None], name="Old_value")
        self.returns = tf.placeholder(tf.float32, [None], name="Returns")
        self.prevneglogp = tf.placeholder(tf.float32, [None], name="Old_pi_a")
        self.lr = tf.placeholder(tf.float32, [], name="Learning_rate")

        with tf.variable_scope('Actor_loss'):
            pi_a = self.network.action.neglogp(self.actions)
            ratio = tf.exp(self.prevneglogp - pi_a)
            actor_loss = ratio * -self.advantage
            clipped_loss = tf.clip_by_value(ratio, 1 - self.epsilon, 1 + self.epsilon) * -self.advantage
            self.actor_loss = tf.reduce_mean(tf.maximum(actor_loss, clipped_loss))
            self.clipfrac = tf.reduce_mean(tf.to_float(tf.greater(tf.abs(ratio - 1.0), self.epsilon)))

        with tf.variable_scope('Entropy'):
            self.entropy = tf.reduce_mean(self.network.action.entropy())
        with tf.variable_scope('Critic_loss'):
            critic_loss1 = tf.squared_difference(self.returns, self.value)
            critic_loss2 = tf.squared_difference(self.returns, self.old_value + tf.clip_by_value(self.value - self.old_value, -self.epsilon, self.epsilon))
            self.vclipfrac = tf.reduce_mean(tf.to_float(tf.greater(tf.abs(self.value - self.old_value), self.epsilon)))
            self.critic_loss = tf.reduce_mean(tf.maximum(critic_loss1, critic_loss2)) * 0.5
        with tf.variable_scope('Total_loss'):
            self.loss =  self.actor_loss - self.entropy * self.ent_coef + self.critic_loss * self.beta
        
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
        tf.summary.scalar('value', tf.reduce_mean(self.value) * tf.sqrt(self.reward_rms._var))
        tf.summary.scalar('clip_fraction', self.clipfrac)
        tf.summary.scalar('learning_rate', self.lr)

        self.summaries = tf.summary.merge_all()
        
    def get_value(self, obs, s=None, m=None):
        feed = {self.network.act_obs : obs}
        if self.network.recurrent:
            feed[self.network.act_state] = s
            feed[self.network.act_mask] = m

        return self.sess.run(self.network.act_value, feed_dict=feed)

    def get_action(self, obs, s=None, m=None):
        feed = {self.network.act_obs : [obs]}
        if self.network.recurrent:
            feed[self.network.act_state] = s
            feed[self.network.act_mask] = [m]
            a, ns = self.sess.run([self.sample, self.network.next_state], feed_dict=feed)
            a = np.squeeze(a, 0)
            return a, ns
        else:
            a = self.sess.run(self.sample, feed_dict=feed)
            a = np.squeeze(a, 0)
            return a

    def get_actions(self, obss, s=None, m=None):
        feed = {self.network.act_obs : obss}
        if self.network.recurrent:
            feed[self.network.act_state] = s
            feed[self.network.act_mask] = m
            a, ns, ap, v = self.sess.run([self.sample, self.network.next_state, self.neglogp, self.network.act_value], feed_dict=feed)
            return a, ap, v, ns
        else:
            a, ap, v = self.sess.run([self.sample, self.neglogp, self.network.act_value], feed_dict=feed)
            return a, ap, v

    def run_trains(self, s_lst, a_lst, returns_lst, advantage_lst, action_prob_lst, value_lst, learning_rate, hs_lst=None, m_lst=None):
        end = 0
        size = self.network.nenvs * self.network.nsteps
        order = np.arange(size)
        minibatch_size = size // self.network.minibatches
        if not self.network.recurrent:
            for e in range(self.epochs):
                np.random.shuffle(order)
                for i in range(0, size, minibatch_size):
                    end = i + minibatch_size
                    ind = order[i:end]
                    slices = (arr[ind] for arr in (s_lst, a_lst, returns_lst, advantage_lst, action_prob_lst, value_lst))
                    islast = e==self.epochs-1 and i==size-minibatch_size
                    self.run_train(*slices, None, learning_rate, None, islast)
        else:
            envsperbatch = self.network.nenvs // self.network.minibatches
            envinds = np.arange(self.network.nenvs)
            flatinds = np.arange(self.network.nenvs * self.network.nsteps).reshape(self.network.nenvs, self.network.nsteps)
            for e in range(self.epochs):
                np.random.shuffle(envinds)
                for i in range(0, self.network.nenvs, envsperbatch):
                    end = i + envsperbatch
                    mbenvinds = envinds[i:end]
                    mbflatinds = flatinds[mbenvinds].ravel()
                    slices = (arr[mbflatinds] for arr in (s_lst, a_lst, returns_lst, advantage_lst, action_prob_lst, value_lst, m_lst))
                    hstates = hs_lst[mbenvinds]
                    islast = e==self.epochs-1 and i==self.network.nenvs-envsperbatch
                    self.run_train(*slices, learning_rate, hstates, islast)
                    

    def run_train(self, s_lst, a_lst, returns_lst, advantage_lst, action_prob_lst, value_lst, m_lst, learning_rate, hs_lst, log=False):
        feed = {self.returns:returns_lst, \
                self.advantage:advantage_lst, self.prevneglogp:action_prob_lst, self.old_value:value_lst, \
                self.actions:a_lst, self.obs : s_lst, self.lr : learning_rate}
        if self.network.recurrent:
            feed[self.network.mask] = m_lst
            feed[self.network.state] = hs_lst
        if log and self.writer != None:
            _, summary = self.sess.run([self.train, self.summaries], feed_dict=feed)
            self.writer.add_summary(summary, self.sess.run(self.global_step))
        else:
            self.sess.run(self.train, feed_dict=feed)

    def calc_gae(self, r_lst, value_lst, done_lst, gamma=None):
        if gamma == None:
            gamma = self.gamma
        cur = 0.0
        size = len(r_lst)
        advantage_lst = np.zeros(size)
        for i in reversed(range(size)):
            delta = r_lst[i] + gamma * value_lst[i+1] * (1-done_lst[i+1]) - value_lst[i]
            advantage_lst[i] = cur = self.lamda * gamma * (1-done_lst[i+1]) * cur + delta
        return advantage_lst

    def train_batches(self, batch_lst, learning_rate=None, hs_lst=None):
        if learning_rate == None:
            learning_rate = self.learning_rate

        curr_step = self.sess.run(self.global_step)

        for i in range(self.network.nenvs):
            size = len(batch_lst[i][1])
            cumulative_lst = np.zeros([size])
            for j in range(size):
                self.cumulative_reward[i] = batch_lst[i][2][j] + self.cumulative_reward[i] * self.gamma
                cumulative_lst[j] = self.cumulative_reward[i]
                self.cumulative_reward[i] *= (1-batch_lst[i][3][j])

            self.reward_rms.update(cumulative_lst)
        
        s_lsts = np.empty(shape=[0, *np.shape(batch_lst[0][0][0])])
        a_lsts = np.empty(shape=[0, *np.shape(batch_lst[0][1][0])])
        advantage_lsts = np.empty([0])
        action_prob_lsts = np.empty([0])
        value_lsts = np.empty([0])
        returns_lsts = np.empty([0])
        m_lst = np.empty([0])
        for batch in batch_lst:
            s_lst = np.asarray(batch[0])
            if self.network.action.type == actiontype.Discrete:
                a_lst = np.asarray(batch[1], dtype=np.int32)
            else:
                a_lst = np.asarray(batch[1], dtype=np.float32)

            r_lst = np.asarray(batch[2], dtype=np.float32)
            r_lst /= math.sqrt(self.reward_rms.var)
            r_lst =  np.clip(r_lst, -5 ,5)

            done_lst = np.asarray(batch[3], dtype=np.int32)

            action_prob_lst = np.asarray(batch[4], dtype=np.float32)

            value_lst = np.asarray(batch[5])
            
            advantage_lst = self.calc_gae(r_lst, value_lst, done_lst)
            value_lst = value_lst[:-1]
            returns_lst = value_lst + advantage_lst
            advantage_lst = (advantage_lst - advantage_lst.mean()) / (advantage_lst.std() + 1e-8)

            s_lsts = np.concatenate((s_lsts, s_lst), axis=0)
            a_lsts = np.concatenate((a_lsts, a_lst), axis=0)
            value_lsts = np.concatenate((value_lsts, value_lst), axis=0)
            action_prob_lsts = np.concatenate((action_prob_lsts, action_prob_lst), axis=0)
            advantage_lsts = np.concatenate((advantage_lsts, advantage_lst), axis=0)
            returns_lsts = np.concatenate((returns_lsts, returns_lst), axis=0)
            if self.network.recurrent:
                m_lst = np.concatenate((m_lst, done_lst[:-1]), axis=0)

        if self.network.recurrent:
            self.run_trains(s_lsts, a_lsts, returns_lsts, advantage_lsts, action_prob_lsts, value_lsts, learning_rate, hs_lst, m_lst)
        else:
            self.run_trains(s_lsts, a_lsts, returns_lsts, advantage_lsts, action_prob_lsts, value_lsts, learning_rate)
        
        self.sess.run(self.increment_global_step, feed_dict={self.step_size : len(s_lsts)})

        
       