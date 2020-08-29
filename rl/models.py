import tensorflow as tf
import numpy as np
from . import actiontype

LOG_WEIGHT = False

class ACnet:
    def __init__(self, obs_shape, policy_fn, nenvs, nsteps, minibatches, action_type, action_size, recurrent=False):
        with tf.variable_scope('network', reuse=tf.AUTO_REUSE):
            train_envs = nenvs // minibatches
            self.train_obs = tf.placeholder(dtype=tf.float32, shape=(nenvs * nsteps // minibatches,)+obs_shape, name="train_obs")
            self.act_obs = tf.placeholder(dtype=tf.float32, shape=(nenvs,)+obs_shape, name="act_obs")
            self.nenvs = nenvs
            self.nsteps = nsteps
            self.minibatches = minibatches
            self.recurrent = recurrent

            if recurrent:
                self.policy_network, _, self.state, self.mask, _ = policy_fn(self.train_obs, train_envs)
                self.act_network, self.next_state, self.act_state, self.act_mask, self.initial_state = policy_fn(self.act_obs, nenvs)
            else:
                self.policy_network = policy_fn(self.train_obs)
                self.act_network = policy_fn(self.act_obs)
            self.value = tf.layers.dense(self.policy_network, 1, kernel_initializer=tf.orthogonal_initializer(), name="Value")
            self.act_value = tf.layers.dense(self.act_network, 1, kernel_initializer=tf.orthogonal_initializer(), name="Value")
            if action_type == actiontype.Continuous:
                self.action = actiontype.continuous(self.policy_network, action_size)
                self.act_action = actiontype.continuous(self.act_network, action_size)
            else:
                self.action = actiontype.discrete(self.policy_network, action_size)
                self.act_action = actiontype.discrete(self.act_network, action_size)

        self.act_value = self.act_value[:, 0]
        self.value = self.value[:, 0]


class RNDnet(ACnet):
    def __init__(self, policy_network, value_network, action_type, action_size, state=None, mask=None):
        super().__init__(policy_network, value_network, action_type, action_size, state, mask)



def add_dense(inputs, output_size, activation=None, kernel_initializer=tf.orthogonal_initializer(np.sqrt(2)), name=""):
    layer = tf.layers.dense(inputs, output_size, activation=activation, kernel_initializer=kernel_initializer, name=name)
    if LOG_WEIGHT:
        with tf.variable_scope(name, reuse=True):
            tf.summary.histogram("kernel", tf.get_variable("kernel"))
            tf.summary.histogram("bias", tf.get_variable("bias"))
    return layer

def add_cnn(inputs, filters, kernel_size, strides=(1,1), padding='valid', data_format='channels_last', activation=tf.orthogonal_initializer(np.sqrt(2)), kernel_initializer=None, name=""):
    layer = tf.layers.conv2d(inputs, filters, kernel_size, strides=strides, padding=padding, data_format=data_format, activation=activation, kernel_initializer=kernel_initializer, name=name)
    if LOG_WEIGHT:
        with tf.variable_scope(name, reuse=True):
            tf.summary.histogram("kernel", tf.get_variable("kernel"))
            tf.summary.histogram("bias", tf.get_variable("bias"))
    return layer

def mlp(nodes=64, depth=2, activation=tf.nn.tanh):
    def fn(input):
        network = add_dense(input, nodes, activation=activation, name=f'dense0')
        for i in range(1, depth):
            network = add_dense(network, nodes, activation=activation, name=f'dense{i}')
        return network
    return fn

def nature_cnn(name='cnn'):
    def fn(input):
        network = add_cnn(input, 32, (8, 8), strides=(4, 4), activation=tf.nn.relu, name=f'cnn0')
        network = add_cnn(network, 64, (4, 4), strides=(2, 2), activation=tf.nn.relu, name=f'cnn1')
        network = add_cnn(network, 64, (3, 3), strides=(1, 1), activation=tf.nn.relu, name=f'cnn2')
        network = tf.contrib.layers.flatten(network)
        network = add_dense(network, 512, activation=tf.nn.relu, name=f'dense0')
        return network
    return fn

def atari_lstm(n_envs, n_hidden, init_scale=1.0, name="atari_lstm"):
    def fn(input):
        network = add_cnn(input, 32, (8, 8), strides=(4, 4), activation=tf.nn.relu, name=f'cnn0')
        network = add_cnn(network, 64, (4, 4), strides=(2, 2), activation=tf.nn.relu, name=f'cnn1')
        network = add_cnn(network, 64, (3, 3), strides=(1, 1), activation=tf.nn.relu, name=f'cnn2')
        network = tf.contrib.layers.flatten(network)
        network, ns, s, m, i = lstm(n_envs, n_hidden, init_scale=init_scale)(network)
        return network, ns, s, m, i
    return fn

def lstm(n_envs, n_hidden, init_scale=1.0, name="lstm"):
    def fn(input_tensor):
        with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
            n_batch = input_tensor.shape[0]
            n_steps = n_batch // n_envs
            h = tf.layers.flatten(input_tensor)
            mask_tensor = tf.placeholder(dtype=tf.float32, shape=[n_batch], name="mask")
            state = tf.placeholder(dtype=tf.float32, shape=[n_envs, n_hidden*2], name="state")

            m = batch_to_seq(mask_tensor, n_envs, n_steps)
            h = batch_to_seq(h, n_envs, n_steps)

            _, n_input = [v.value for v in h[0].get_shape()]

            weight_x = tf.get_variable("wx", [n_input, n_hidden * 4], initializer=tf.orthogonal_initializer(init_scale))
            weight_h = tf.get_variable("wh", [n_hidden, n_hidden * 4], initializer=tf.orthogonal_initializer(init_scale))
            bias = tf.get_variable("b", [n_hidden * 4], initializer=tf.constant_initializer(0.0))

            cell_state, hidden = tf.split(axis=1, num_or_size_splits=2, value=state)

            for idx, (_input, mask) in enumerate(zip(h, m)):
                cell_state = cell_state * (1 - mask)
                hidden = hidden * (1 - mask)
                gates = tf.matmul(_input, weight_x) + tf.matmul(hidden, weight_h) + bias
                in_gate, forget_gate, out_gate, cell_candidate = tf.split(axis=1, num_or_size_splits=4, value=gates)
                in_gate = tf.nn.sigmoid(in_gate)
                forget_gate = tf.nn.sigmoid(forget_gate)
                out_gate = tf.nn.sigmoid(out_gate)
                cell_candidate = tf.tanh(cell_candidate)
                cell_state = forget_gate * cell_state + in_gate * cell_candidate
                hidden = out_gate * tf.tanh(cell_state)
                h[idx] = hidden
            
            initial_state = np.zeros(state.shape.as_list(), dtype=float)
            cell_state_hidden = tf.concat(axis=1, values=[cell_state, hidden])

            h = seq_to_batch(h)

        return h, cell_state_hidden, state, mask_tensor, initial_state
    return fn


def batch_to_seq(tensor_batch, n_envs, n_steps):
    tensor_batch = tf.reshape(tensor_batch, [n_envs, n_steps, -1])
    return [tf.squeeze(v, [1]) for v in tf.split(axis=1, num_or_size_splits=n_steps, value=tensor_batch)]


def seq_to_batch(tensor_sequence):
    shape = tensor_sequence[0].get_shape().as_list()
    assert len(shape) > 1
    n_hidden = tensor_sequence[0].get_shape()[-1].value
    return tf.reshape(tf.concat(axis=1, values=tensor_sequence), [-1, n_hidden])
