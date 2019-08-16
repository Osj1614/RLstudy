"""
Simple Asynchronous Methods for Deep Reinforcement Learning (A3C)
- It mimics A3C by using multi threads
- Distributed Tensorflow is preferred because of Python's GIL
"""
import tensorflow as tf
import numpy as np
import gym
import os
from AtariAgent import AtariAgent

def main():
    tf.reset_default_graph()
    sess = tf.InteractiveSession()

    checkpoint_dir = "checkpoint_ppo"
    monitor_dir = "monitors_ppo"
    save_path = os.path.join(checkpoint_dir, "model.ckpt")

    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
        print("Directory {} was created".format(checkpoint_dir))

    input_shape = [4, 84, 84]

    env = gym.make("PongDeterministic-v4")
    env = gym.wrappers.Monitor(env, monitor_dir, force=True)

    output_dim = 3
    coord = tf.train.Coordinator()

    Atari_Agent = AtariAgent(env=env,
                            session=sess,
                            num_episodes=1000,
                            name="global",
                            coord=coord,
                            input_shape=input_shape,
                            output_dim=output_dim
                            )

    init = tf.global_variables_initializer()
    sess.run(init)
    if tf.train.get_checkpoint_state(os.path.dirname(save_path)):
        var_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, "global")
        saver = tf.train.Saver(var_list=var_list)
        saver.restore(sess, save_path)
        print("Model restored to global")

    else:
        print("No model is found")
    agents = [AtariAgent(sess, gym.make("PongDeterministic-v4"), 100, "{} agent".format(i), coord, network=Atari_Agent.network) for i in range(9)]
    agents.append(Atari_Agent)
    for aa in agents:
        aa.start()

    coord.join(agents)
    print("All Episode finished")

    var_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, "global")
    saver = tf.train.Saver(var_list=var_list)
    saver.save(sess, save_path)
    print('Checkpoint Saved to {}'.format(save_path))

    print("Closing environment")
    env.close()

    sess.close()
    


if __name__ == '__main__':
    main()