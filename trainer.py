
import tensorflow as tf
import os
import gym
from runner import Runner
from procrunner import ProcRunner

def load_model(sess, model, save_path):
    var_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, model.name)
    saver = tf.train.Saver(var_list=var_list)

    if tf.train.get_checkpoint_state(os.path.dirname(save_path)):
        saver.restore(sess, save_path)
        print("Model restored to global")
    else:
        print("No model is found")

    if not os.path.exists("models/" + model.name):
        os.makedirs("models/" + model.name)
        print("Directory models/" + model.name + " was created")

    return saver

def train(sess, model, env_name, num_steps, update_interval, num_envs=1, atari=False):
    model.make_summary()
    summaries = tf.summary.merge_all()
    writer = tf.summary.FileWriter("./logs/" + model.name, sess.graph)
    with tf.variable_scope(model.name):
        global_step = tf.Variable(initial_value=0, trainable=False, name="global_step")
        increment_global_step = tf.assign_add(global_step, update_interval, name = 'increment_global_step')
    save_path = "models/" + model.name + "/model.ckpt"
    sess.run(tf.global_variables_initializer())
    saver = load_model(sess, model, save_path)

    currstep = sess.run(global_step)
    
    if num_envs == 1:
        runner = Runner(gym.make(env_name), update_interval, writer=writer, clip=True)
    else:
        runner = ProcRunner(env_name, num_envs, update_interval, writer=writer, atari=atari)

    for i in range((num_steps - currstep) // update_interval):
        sess.run(increment_global_step)
        currstep += update_interval

        batches = runner.run_steps(model, currstep)
        
        lr = model.learning_rate * (1 - currstep / num_steps) #LR annealing
        #lr = model.learning_rate
        if lr <= 1e-8:
            lr = 1e-8
        summary_data = model.train_batches(batches, lr, summaries)
        writer.add_summary(summary_data, currstep)

        if i % 50 == 0:
            saver.save(sess, save_path)
    if num_envs > 1:
        runner.close()
    saver.save(sess, save_path)

def run_only(sess, model, env):
    save_path = "models/" + model.name + "/model.ckpt"
    load_model(sess, model, save_path)
    total_reward = 0
    env.render()
    runner = Runner(env, 0, True)
    while True:
        total_reward = runner.playgame(model)
        print(total_reward)