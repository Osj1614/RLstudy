import tensorflow as tf
import numpy as np

def save_session(save_path, sess, scope=None):
    f = open(save_path, 'w')
    for var in tf.global_variables(scope):
        values = sess.run(var)
        f.write(f"{var.name}\n{values.shape}\n")
        for value in np.reshape(values, [-1]):
            f.write(f"{value} ")
        f.write('\n')
    f.close()


