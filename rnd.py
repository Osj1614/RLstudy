import numpy as np
import math
from ppo import PPO
import tensorflow as tf
import actiontype
from running_std import RunningMeanStd

class RND(PPO):
    def __init__(self, sess, state, state_rms, network, action_type, action_size, target_network, predictor_network, value_in_network, \
            value_network=None, name="", learning_rate=0.00025, beta=0.5, beta2=0.01, gamma=0.99, epsilon=0.1, lamda=0.95, max_grad_norm=0.5, epochs=4, minibatch_size=16, \
                gamma_in=0.99, coef_in=0.5, rnd_lr=0.00025, init_runs=100):
        self.gamma_in = gamma_in
        self.rnd_lr = rnd_lr
        self.coef_in = coef_in
        self.target_network = target_network
        self.predictor_network = predictor_network
        self.reward_in = tf.reduce_mean(tf.square(self.predictor_network - self.target_network), axis=-1)
        self.reward_in_rms = RunningMeanStd(sess, scope="cumulative_reward_in")
        self.cumulative_reward_in = None
        self.state_rms = state_rms
        self.rnd_lr = rnd_lr
        self.value_in = tf.layers.dense(value_in_network, 1, kernel_initializer=tf.orthogonal_initializer(), name="Value_in")
        super().__init__(sess, state, network, action_type, action_size, value_network, name, learning_rate, beta, beta2, gamma, epsilon, lamda, max_grad_norm, epochs, minibatch_size)

    def make_summary(self):
        tf.summary.scalar('critic_in_loss', self.critic_in_loss)
        tf.summary.scalar('value_in', tf.reduce_mean(self.value_in))
        tf.summary.scalar('rnd_loss', self.rnd_loss)
        tf.summary.scalar('intrinsic_reward', tf.reduce_mean(self.reward_in) / tf.sqrt(self.reward_in_rms._var))
        super().make_summary()

    def bulid_train(self, network, value_network=None):
        self.advantage = tf.placeholder(tf.float32, [None], name="Advantage")
        self.old_value = tf.placeholder(tf.float32, [None], name="Old_value")
        self.returns = tf.placeholder(tf.float32, [None], name="Returns")
        self.returns_in = tf.placeholder(tf.float32, [None], name="Returns_intrinsic")
        self.prevneglogp = tf.placeholder(tf.float32, [None], name="Old_pi_a")
        self.lr = tf.placeholder(tf.float32, [], name="Learning_rate")

        if value_network == None:
            self.value = tf.layers.dense(network, 1, kernel_initializer=tf.orthogonal_initializer(), name="Value")
        else:
            self.value = tf.layers.dense(value_network, 1, kernel_initializer=tf.orthogonal_initializer(), name="Value")
        self.value = self.value[:, 0]
        self.value_in = self.value_in[:, 0]

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
            critic_loss2 = tf.squared_difference(self.returns, self.old_value + tf.clip_by_value(self.value - self.old_value, -self.epsilon, self.epsilon))
            self.vclipfrac = tf.reduce_mean(tf.to_float(tf.greater(tf.abs(self.value - self.old_value), self.epsilon)))
            self.critic_loss = tf.reduce_mean(tf.maximum(critic_loss1, critic_loss2)) * 0.5

            self.critic_in_loss = tf.reduce_mean(tf.squared_difference(self.returns_in, self.value_in)) * 0.5
            
        with tf.variable_scope('RND_loss'):
            self.rnd_loss = tf.reduce_mean(tf.square(tf.stop_gradient(self.target_network) - self.predictor_network))
        
        with tf.variable_scope('Total_loss'):
            self.loss =  self.actor_loss - self.entropy * self.beta2 + (self.critic_loss + self.critic_in_loss) * self.beta
        
        params = tf.trainable_variables(self.name)
        with tf.variable_scope('train'):
            trainer = tf.train.AdamOptimizer(learning_rate=self.lr, epsilon=1e-5)
            grads_and_var = trainer.compute_gradients(self.loss, params)
            grads, var = zip(*grads_and_var)
            if self.max_grad_norm != None:
                grads, _ = tf.clip_by_global_norm(grads, self.max_grad_norm)
            grads_and_var = list(zip(grads, var))
            self.train = trainer.apply_gradients(grads_and_var)

        with tf.variable_scope('train_rnd'):
            self.train_rnd = tf.train.AdamOptimizer(learning_rate=self.rnd_lr).minimize(self.rnd_loss)

    def get_actions(self, states):
        a, ap = self.sess.run([self.sample, self.neglogp], feed_dict={self.state : states})
        return a, ap

    def get_intrinsic(self, state):
        return self.sess.run((self.reward_in, self.value_in), feed_dict={self.state : state})

    def run_trains(self, s_lst, a_lst, returns_lst, advantage_lst, action_prob_lst, value_lst, returns_in_lst, learning_rate):
        end = 0
        size = len(returns_lst)
        order = np.arange(size)
        for _ in range(self.epochs):
            np.random.shuffle(order)
            for i in range(0, size, self.minibatch_size):
                end = i + self.minibatch_size
                if end <= size:
                    ind = order[i:end]
                    slices = (arr[ind] for arr in (s_lst, a_lst, returns_lst, advantage_lst, action_prob_lst, value_lst, returns_in_lst))
                    self.run_train(*slices, learning_rate)
        self.run_train_rnd(s_lst)

    def run_train(self, s_lst, a_lst, returns_lst, advantage_lst, action_prob_lst, value_lst, returns_in_lst, learning_rate):
        self.sess.run(self.train, feed_dict={self.returns:returns_lst, \
            self.advantage:advantage_lst, self.prevneglogp:action_prob_lst, self.old_value:value_lst, \
            self.actions:a_lst, self.state : s_lst, self.returns_in : returns_in_lst, self.lr : learning_rate})

    def run_train_rnd(self, s_lst):
        for i in range(0, len(s_lst), self.minibatch_size):
            end = i + self.minibatch_size
            self.sess.run(self.train_rnd, feed_dict={self.state : s_lst[i:end]})

    def update_rms(self, batch_lst):
        if not isinstance(self.cumulative_reward, list) or len(self.cumulative_reward) != len(batch_lst):
            self.cumulative_reward = [0 for _ in range(len(batch_lst))]
        if not isinstance(self.cumulative_reward_in, list) or len(self.cumulative_reward_in) != len(batch_lst):
            self.cumulative_reward_in = [0 for _ in range(len(batch_lst))]
            
        r_in_lsts = list()
        value_in_lsts = list()

        for i in range(len(batch_lst)):
            size = len(batch_lst[i][1])
            r_in_lst, value_in_lst = self.get_intrinsic(batch_lst[i][0])
            r_in_lst = r_in_lst[1:]
            r_in_lsts.append(r_in_lst)
            value_in_lsts.append(value_in_lst)
            cumulative_lst = np.zeros([size])
            cumulative_in_lst = np.zeros([size])
            for j in range(size):
                self.cumulative_reward_in[i] = r_in_lst[j] + self.cumulative_reward_in[i] * self.gamma_in
                self.cumulative_reward[i] = batch_lst[i][2][j] + self.cumulative_reward[i] * self.gamma
                cumulative_lst[j] = self.cumulative_reward[i]
                cumulative_in_lst[j] = self.cumulative_reward_in[i]
                self.cumulative_reward[i] *= batch_lst[i][3][j]
            self.reward_in_rms.update(cumulative_in_lst)
            self.reward_rms.update(cumulative_lst)
            self.state_rms.update(batch_lst[i][0])
        return r_in_lsts, value_in_lsts

    def train_batches(self, batch_lst, learning_rate=None, writer=None):
        if learning_rate == None:
            learning_rate = self.learning_rate

        curr_step = self.sess.run(self.global_step)

        if curr_step < 10000:
            for batch in batch_lst:
                self.run_train_rnd(batch[0])
            self.sess.run(self.increment_global_step, feed_dict={self.step_size : len(batch_lst) * len(batch_lst[0][1])})
            return

        r_in_lsts, value_in_lsts = self.update_rms(batch_lst)

        s_lsts = np.empty(shape=[0, *np.shape(batch_lst[0][0][0])])
        a_lsts = np.empty(shape=[0, *np.shape(batch_lst[0][1][0])])
        advantage_lsts = np.empty([0])
        action_prob_lsts = np.empty([0])
        value_lsts = np.empty([0])
        returns_lsts = np.empty([0])
        returns_in_lsts = np.empty([0])
        i = 0
        for batch in batch_lst:
            value_in_lst = value_in_lsts[i]
            r_in_lst = r_in_lsts[i]
            r_in_lst /= math.sqrt(self.reward_in_rms.var)
            r_in_lst =  np.clip(r_in_lst, 0 ,5)
            s_lst = np.asarray(batch[0])
            value_lst = self.get_value(s_lst)
            s_lst = s_lst[:-1]
            if self.action_type == actiontype.Discrete:
                a_lst = np.asarray(batch[1], dtype=np.int32)
            else:
                a_lst = np.asarray(batch[1], dtype=np.float32)
            r_lst = np.asarray(batch[2], dtype=np.float32)
            r_lst /= math.sqrt(self.reward_rms.var)
            r_lst =  np.clip(r_lst, -5 ,5)
            done_lst = np.asarray(batch[3], dtype=np.int32)
            action_prob_lst = np.asarray(batch[4], dtype=np.float32)
            size = len(done_lst)

            advantage_lst = np.zeros([size])

            advantage_lst = self.calc_gae(r_lst, value_lst, done_lst)
            value_lst = value_lst[:-1]
            returns_lst = value_lst + advantage_lst

            advantage_in_lst = self.calc_gae(r_in_lst, value_in_lst, np.ones_like(done_lst), self.gamma_in)
            value_in_lst = value_in_lst[:-1]
            returns_in_lst = value_in_lst + advantage_in_lst

            advantage_lst = advantage_lst+advantage_in_lst*self.coef_in
            advantage_lst = (advantage_lst - advantage_lst.mean()) / (advantage_lst.std() + 1e-8)

            s_lsts = np.concatenate((s_lsts, s_lst), axis=0)
            a_lsts = np.concatenate((a_lsts, a_lst), axis=0)
            value_lsts = np.concatenate((value_lsts, value_lst), axis=0)
            action_prob_lsts = np.concatenate((action_prob_lsts, action_prob_lst), axis=0)
            advantage_lsts = np.concatenate((advantage_lsts, advantage_lst), axis=0)
            returns_lsts = np.concatenate((returns_lsts, returns_lst), axis=0)
            returns_in_lsts = np.concatenate((returns_in_lsts, returns_in_lst), axis=0)
            i += 1

        self.run_trains(s_lsts, a_lsts, returns_lsts, advantage_lsts, action_prob_lsts, value_lsts, returns_in_lsts, learning_rate)
        
        self.sess.run(self.increment_global_step, feed_dict={self.step_size : len(s_lsts)})

        if writer != None:
            summary_data = self.sess.run(self.summaries, feed_dict={self.returns:returns_lsts, self.actions:a_lsts, self.advantage:advantage_lsts, \
                self.prevneglogp:action_prob_lsts, self.old_value:value_lsts, self.state : s_lsts, self.returns_in : returns_in_lsts, self.lr : learning_rate})
            writer.add_summary(summary_data, curr_step)
        
       