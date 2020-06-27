import tensorflow as tf
import numpy as np


Continuous = 1
Discrete = 2

class actiontype:
    def neglogp(self, x):
        pass

    def entropy(self, x):
        pass

    def sample(self):
        pass

class continuous(actiontype):
    def __init__(self, network, size):
        self.mean = tf.layers.dense(network, size, kernel_initializer=tf.orthogonal_initializer(), name="policy/mean")
        self.size = size
        self.type = Continuous
        self.logstd = tf.get_variable(name='policy/logstd', shape=[1, self.size], initializer=tf.zeros_initializer())
        self.std = tf.exp(self.logstd)
    
    def neglogp(self, x):
        return 0.5 * tf.reduce_sum(tf.square((x - self.mean) / self.std), axis=-1) + 0.5 * np.log(2.0 * np.pi) * self.size + tf.reduce_sum(self.logstd, axis=-1)

    def entropy(self):
        return tf.reduce_sum(self.logstd + .5 * np.log(2.0 * np.pi * np.e), axis=-1)

    def sample(self):
        return self.mean + self.std * tf.random_normal(tf.shape(self.mean))

class discrete(actiontype):
    def __init__(self, network, size):
        self.policy = tf.layers.dense(network, size, kernel_initializer=tf.orthogonal_initializer(), name="policy")
        self.size = size
        self.type = Discrete

    def neglogp(self, x):
        return tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.policy, labels=tf.one_hot(x, self.size))

    def entropy(self):
        a0 = self.policy - tf.reduce_max(self.policy, axis=-1, keepdims=True)
        ea0 = tf.exp(a0)
        z0 = tf.reduce_sum(ea0, axis=-1, keepdims=True)
        p0 = ea0 / z0
        return tf.reduce_sum(p0 * (tf.log(z0) - a0), axis=-1)
    
    def sample(self):
        u = tf.random_uniform(tf.shape(self.policy), dtype=self.policy.dtype)
        return tf.argmax(self.policy - tf.log(-tf.log(u)), axis=-1)
        #http://amid.fish/humble-gumbel
