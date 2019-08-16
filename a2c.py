import numpy as np
import tensorflow as tf

learning_rate = 0.0005
beta = 0.5
gamma = 0.98
epsilon = 0.1
lamda = 0.95

class ActorCritic:
    def __init__(self, sess, input, network, action_size):
        self.input = input
        self.sess = sess
        self.action_size = action_size
        self.bulidNetwork(network)
        
    def bulidNetwork(self, network):
        self.Value_expect = tf.placeholder(tf.float32, [None])
        self.Action = tf.placeholder(tf.int32, [None])
        self.Value = tf.layers.dense(network, 1)
        self.Policy = tf.layers.dense(network, self.action_size, activation=tf.nn.softmax)

        delta = self.Value_expect - tf.reshape(self.Value, [-1])
        pi_a = self.Policy * tf.one_hot(self.Action, self.action_size)
        pi_a = tf.reduce_sum(pi_a, 1)
        self.loss = -(tf.stop_gradient(delta) * tf.log(pi_a)) + beta * delta * delta
        self.train = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(self.loss)

    def getValue(self, input):
        return self.Value.eval(feed_dict={self.input : input}, session=self.sess)

    def getPi(self, input):
        return self.Policy.eval(feed_dict={self.input : input}, session=self.sess)

    def getAction(self, input):
        prob = np.squeeze(self.getPi([input]))
        choice = np.random.choice(np.arange(len(prob)), p=prob.ravel())
        return choice, prob[choice]

    def TrainBatch(self, s_lst, a_lst, r_lst, done_lst, _, __):
        s2_lst = s_lst[1:]
        s_lst = s_lst[:-1]
        expect_lst = np.reshape(r_lst + gamma * np.reshape(self.getValue(s2_lst), [-1]) * done_lst, [-1])
        self.sess.run(self.train, feed_dict={self.input:s_lst, self.Value_expect:expect_lst, self.Action:a_lst})