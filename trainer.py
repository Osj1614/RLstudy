
import tensorflow as tf
import os
import gym
import time
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

def train(sess, model, env_name, num_steps, update_interval, log_interval=10, save_interval=50, num_envs=1, atari=False):
    model.make_summary()

    summaries = tf.summary.merge_all()
    writer = tf.summary.FileWriter("./logs/" + model.name, sess.graph)
    with tf.variable_scope(model.name):
        global_step = tf.Variable(initial_value=0, trainable=False, name="global_step")
        increment_global_step = tf.assign_add(global_step, update_interval*num_envs, name = 'increment_global_step')
    save_path = "models/" + model.name + "/model.ckpt"
    sess.run(tf.global_variables_initializer())
    saver = load_model(sess, model, save_path)

    currstep = sess.run(global_step)
    
    if num_envs == 1:
        runner = Runner(gym.make(env_name), update_interval)
    else:
        runner = ProcRunner(env_name, num_envs, update_interval, atari=atari)

    total_iter = int(num_steps // (update_interval * num_envs))
    curr_iter = currstep // (update_interval * num_envs)
    prevtime = time.time()
    for i in range(curr_iter, total_iter+1):
        sess.run(increment_global_step)
        currstep += update_interval * num_envs

        batches = runner.run_steps(model, currstep)
        
        if (i+1) % log_interval == 0:
            avg, high = runner.get_avg_high()
            print(f"Average score:\t{round(avg,3)}")
            print(f"High score:\t{round(high,3)}")
            print(f"progress:\t{i+1}/{total_iter+1} ({round((i+1)/(total_iter+1)*100, 2)}%)")
            currtime = time.time()
            time_passed = currtime - prevtime
            print(f"elapsed time:\t{round(time_passed, 3)} second")
            print(f"time left:\t{round(time_passed*(total_iter-i)/log_interval/3600, 3)} hour")
            prevtime = currtime
            if high != -100000:
                score_summary_data = tf.Summary(value=[tf.Summary.Value(tag="score", simple_value=avg)])
                writer.add_summary(score_summary_data, currstep)
                score_summary_data = tf.Summary(value=[tf.Summary.Value(tag="score_high", simple_value=high)])
                writer.add_summary(score_summary_data, currstep)
            print('-----------------------------------------------------------')


        if callable(model.learning_rate):
            lr = model.learning_rate(currstep / num_steps)
        else:
            lr = model.learning_rate
        if lr <= 1e-8:
            lr = 1e-8
        summary_data = model.train_batches(batches, lr, summaries)
        if summary_data != None:
            writer.add_summary(summary_data, currstep)

        if (i+1) % save_interval == 0:
            saver.save(sess, save_path)
        if i % (total_iter // 4) == 0 and i != 0:
            saver.save(sess, f"models/{model.name}/{i // (total_iter // 4)}/model.ckpt")

    if num_envs > 1:
        runner.close()
    saver.save(sess, save_path)

def run_only(sess, model, env, render=True):
    save_path = "models/" + model.name + "/model.ckpt"
    load_model(sess, model, save_path)
    total_reward = 0
    env.render()
    runner = Runner(env, 0)
    avg = 0
    high = -1000
    for i in range(100):
        total_reward = runner.playgame(model, render=render)
        print(total_reward)
        if total_reward > high:
            high = total_reward
        avg += total_reward
    print(f"average: {avg/100}")
    print(f"max score: {high}")
