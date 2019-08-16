import numpy as np
import tensorflow as tf

learning_rate = 0.001
beta = 1
beta2 = 0.00
gamma = 0.99
epsilon = 0.2
lamda = 0.95

class PPO:
    def __init__(self, sess, state, network, action_size):
        self.state = state
        self.sess = sess
        self.action_size = action_size
        self.bulidNetwork(network)
        
    def bulidNetwork(self, network):
        self.Advantage = tf.placeholder(tf.float32, [None], name="Advantage")
        self.td_target = tf.placeholder(tf.float32, [None], name="td_target")
        self.Action = tf.placeholder(tf.int32, [None], name="Action")
        self.Old_pi_a = tf.placeholder(tf.float32, [None], name="Old_pi_a")
        self.Value = tf.layers.dense(network, 1, name="Value")
        self.Policy = tf.layers.dense(network, self.action_size, activation=tf.nn.softmax, name="Policy")

        pi_a = self.Policy * tf.one_hot(self.Action, self.action_size)
        pi_a = tf.reduce_sum(pi_a, 1)

        ratio = tf.exp(tf.log(pi_a) - tf.log(self.Old_pi_a))

        entropy = self.Policy * tf.log(self.Policy)
        entropy = tf.reduce_sum(entropy, 1)
        
        actor_loss = -tf.minimum(ratio * self.Advantage, tf.clip_by_value(ratio, 1 - epsilon, 1 + epsilon) * self.Advantage)
        critic_loss = tf.squared_difference(self.td_target, tf.squeeze(self.Value))

        self.loss =  tf.reduce_sum(actor_loss + entropy * beta2 + critic_loss * beta)
        self.train = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(self.loss)

    def getValue(self, state):
        feed_dict = {self.state : state}
        return self.sess.run(self.Value, feed_dict)

    def getPi(self, state):
        feed_dict = {self.state : state}
        return self.sess.run(self.Policy, feed_dict)

    def getAction(self, state):
        prob = np.squeeze(self.getPi([state]))
        choice = np.random.choice(np.arange(len(prob)), p=prob.ravel())
        return choice, prob[choice]

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
            self.sess.run(self.train, feed_dict={self.state : s_lst, self.td_target:td_target_lst, self.Action:a_lst, self.Advantage:advantage_lst, self.Old_pi_a:action_prob_lst})