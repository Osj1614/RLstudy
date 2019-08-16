import numpy as np
import math
import tensorflow as tf

learning_rate = 0.0005
beta = 0.5
beta2 = 0.01
gamma = 0.98
epsilon = 0.2
lamda = 0.95

class ContinuousPPO:
    def __init__(self, sess, state, network, action_size):
        self.state = state
        self.sess = sess
        self.action_size = action_size
        self.bulidNetwork(network)
        
    def get_log_prob(self, value, mean, var):
        return -((mean - value) ** 2 / (2*var)) - math.log(math.sqrt(2*math.pi*var))
    
    def bulidNetwork(self, network):
        self.advantage = tf.placeholder(tf.float32, [None], name="Advantage")
        self.td_target = tf.placeholder(tf.float32, [None], name="td_target")
        self.action = tf.placeholder(tf.float32, [None, self.action_size], name="Action")
        self.log_old_pi_a = tf.placeholder(tf.float32, [None], name="log_old_pi_a")
        self.value = tf.layers.dense(network, 1, name="Value")
        self.policy_mean = tf.layers.dense(network, self.action_size, activation=tf.nn.tanh, name="Policy_mean")
        self.policy_var = tf.layers.dense(network, self.action_size, activation=tf.nn.softplus, name="Policy_var")

        log_pi_a = -tf.reduce_sum((self.policy_mean - self.action) ** 2 / (2*self.policy_var) - tf.log(tf.sqrt(2 * math.pi * self.policy_var)), 1)

        ratio = tf.exp(log_pi_a - self.log_old_pi_a)
        
        entropy = -tf.reduce_sum(tf.log(tf.sqrt(2 * math.pi * math.e * self.policy_var)), 1)

        actor_loss = -tf.minimum(ratio * self.advantage, tf.clip_by_value(ratio, 1 - epsilon, 1 + epsilon) * self.advantage)
        critic_loss = tf.square(self.td_target - tf.reshape(self.value, [-1]))

        self.loss = tf.reduce_sum(actor_loss + entropy * beta2 + critic_loss * beta)
        self.train = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(self.loss)

    def getValue(self, state):
        feed_dict = {self.state : state}
        return self.sess.run(self.value, feed_dict=feed_dict)

    def getPi(self, state):
        feed_dict = {self.state : state}
        return self.sess.run([self.policy_mean, self.policy_var], feed_dict=feed_dict)

    def getAction(self, state):
        mean, var = self.getPi([state])
        mean = np.squeeze(mean, 0)
        var = np.squeeze(var, 0)
        choices = np.random.normal(mean, var)
        return choices, np.sum(self.get_log_prob(choices, mean, var))

    def TrainBatch(self, s_lst, a_lst, r_lst, done_lst, action_prob_lst, epochs):
        s2_lst = np.array(s_lst[1:])
        s_lst = np.array(s_lst[:-1])
        r_lst = np.array(r_lst)
        a_lst = np.array(a_lst)
        done_lst = np.array(done_lst)
        action_prob_lst = np.array(action_prob_lst)

        td_target_lst = r_lst + gamma * np.reshape(self.getValue(s2_lst), [-1]) * done_lst
        value_lst = np.reshape(self.getValue(s_lst), [-1])
        advantage_lst = np.zeros_like(td_target_lst, dtype=np.float32)
        cur = 0
        for i in reversed(range(len(advantage_lst))):
            cur = cur * gamma * lamda + (td_target_lst[i] - value_lst[i])
            advantage_lst[i] = cur

        for _ in range(epochs):
            self.sess.run(self.train, feed_dict={self.state : s_lst, self.td_target:td_target_lst, self.action:a_lst, self.advantage:advantage_lst, self.log_old_pi_a:action_prob_lst})
