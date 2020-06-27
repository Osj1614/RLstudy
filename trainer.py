
import tensorflow as tf
import os
import gym
import time
from gym.spaces import Box
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

def train(sess, model, env_name, num_steps, log_interval=10, save_interval=50, atari=False):
    writer = tf.summary.FileWriter("./logs/" + model.name, sess.graph)
    save_path = "models/" + model.name + "/model.ckpt"
    sess.run(tf.global_variables_initializer())
    saver = load_model(sess, model, save_path)
    num_envs = model.network.nenvs
    update_interval = model.network.nsteps
    currstep = sess.run(model.global_step)
    if num_envs == 1:
        runner = Runner(model, gym.make(env_name))
    else:
        runner = ProcRunner(model, env_name, atari=atari)

    total_iter = int(num_steps // (update_interval * num_envs))
    curr_iter = currstep // (update_interval * num_envs)
    prevtime = time.time()
    for i in range(curr_iter, total_iter+1):
        currstep += update_interval * num_envs
        if model.network.recurrent:
            batches, hs = runner.run_steps(currstep)
        else:
            batches = runner.run_steps(currstep)
        
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
                score_summary_data = tf.Summary(value=[tf.Summary.Value(tag=f"{model.name}/score", simple_value=avg)])
                writer.add_summary(score_summary_data, currstep)
                score_summary_data = tf.Summary(value=[tf.Summary.Value(tag=f"{model.name}/score_high", simple_value=high)])
                writer.add_summary(score_summary_data, currstep)
            print('-----------------------------------------------------------')


        if callable(model.learning_rate):
            lr = model.learning_rate(currstep / num_steps)
        else:
            lr = model.learning_rate
        if lr <= 1e-8:
            lr = 1e-8

        if model.network.recurrent:
            model.train_batches(batches, lr, writer, hs)
        else:
            model.train_batches(batches, lr, writer)

        if (i+1) % save_interval == 0:
            saver.save(sess, save_path)
        if i % (total_iter // 4) == 0 and i != 0:
            saver.save(sess, f"models/{model.name}/{i // (total_iter // 4)}/model.ckpt")

    if num_envs > 1:
        runner.close()
    saver.save(sess, save_path)

def run_only(sess, model, env, cnt=100, render=True):
    save_path = "models/" + model.name + "/model.ckpt"
    load_model(sess, model, save_path)
    total_reward = 0
    env.render()
    runner = Runner(env, 0)
    avg = 0
    high = -1000
    clip = isinstance(env.action_space, Box)
    for i in range(cnt):
        s = env.reset()
        hs = model.network.initial_state
        total_reward = 0
        while True:
            if render:
                env.render()
            if model.recurrent:
                action, hs = model.get_action(s, hs, False)
            else:
                action = model.get_action(s)

            if clip:
                ns, reward, done, _ = env.step(np.clip(action, env.action_space.low, env.action_space.high))
            else:
                ns, reward, done, _ = env.step(action)
            s = ns
            reward_sum += reward
            if done:
                break
            time.sleep(0.008)
        print(total_reward)
        if total_reward > high:
            high = total_reward
        avg += total_reward
    print(f"average: {avg/100}")
    print(f"max score: {high}")
