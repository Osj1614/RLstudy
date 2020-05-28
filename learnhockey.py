import numpy as np
import airhockey
from hockeyrunner import Runner, play_game, play_human
import tensorflow as tf
import numpy as np
import os
import time
from ppo import PPO
import actiontype
from save_weight import save_session

LOG_WEIGHT = False
DOTEST = True

class randommodel:
    def get_action(self, state):
        return np.array([0, 0]), 1, 0
    
    def get_value(self, state):
        return 0

def add_dense(inputs, output_size, activation=None, kernel_initializer=None, name=""):
    layer = tf.layers.dense(inputs, output_size, activation=activation, kernel_initializer=kernel_initializer, name=name)
    if LOG_WEIGHT:
        with tf.variable_scope(name, reuse=True):
            tf.summary.histogram("kernel", tf.get_variable("kernel"))
            tf.summary.histogram("bias", tf.get_variable("bias"))
    return layer

def load_model(sess, model, save_path):
    var_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, model.name)
    saver = tf.train.Saver(var_list=var_list, max_to_keep=None)

    if tf.train.get_checkpoint_state(os.path.dirname(save_path)):
        saver.restore(sess, save_path)
        print("Model restored to global")
    else:
        print("No model is found")

    if not os.path.exists("models/" + model.name):
        os.makedirs("models/" + model.name)
        print("Directory models/" + model.name + " was created")

    return saver

def traintwo(sess, model, model2, env_count, num_steps, update_interval, log_interval=10, save_interval=50):
    model.make_summary()
    summaries = tf.summary.merge_all()
    #summaries = None
    writer = tf.summary.FileWriter("./logs/" + model.name, sess.graph)
    with tf.variable_scope(model.name):
        global_step = tf.Variable(initial_value=0, trainable=False, name="global_step")
        increment_global_step = tf.assign_add(global_step, update_interval*env_count, name = 'increment_global_step')
    save_path = "models/" + model.name + "/model.ckpt"
    save_path2 = "models/" + model2.name + "/model.ckpt"
    sess.run(tf.global_variables_initializer())
    
    saver = load_model(sess, model, save_path)
    saver2 = load_model(sess, model2, save_path2)
    
    currstep = sess.run(global_step)
    runner = Runner(env_count, update_interval)
    total_iter = num_steps // (update_interval*env_count)
    curr_iter = currstep // (update_interval*env_count)
    prevtime = time.time()
    for i in range(curr_iter, total_iter+1):
        sess.run(increment_global_step)
        currstep += update_interval*env_count

        batch1, batch2 = runner.run_steps(model, model2, currstep)

        if (i+1) % log_interval == 0:
            print(f"progress:\t{i+1}/{total_iter+1} ({round((i+1)/(total_iter+1)*100, 2)}%)")
            currtime = time.time()
            time_passed = currtime - prevtime
            print(f"elapsed time:\t{round(time_passed, 3)} second")
            print(f"time left:\t{round(time_passed*(total_iter-i)/log_interval/3600, 3)} hour")
            prevtime = currtime
            print('-----------------------------------------------------------')

        if callable(model.learning_rate):
            lr = model.learning_rate(currstep / num_steps)
        else:
            lr = model.learning_rate
        if lr <= 1e-8:
            lr = 1e-8
        summary_data = model.train_batches(batch1, lr, summaries)
        model2.train_batches(batch2, lr)
        if summary_data != None:
            writer.add_summary(summary_data, currstep)

        if (i+1) % save_interval == 0:
            saver.save(sess, save_path)
            saver2.save(sess, save_path2)
        if i % (total_iter // 10) == 0 and i != 0:
            saver.save(sess, f"models/{model.name}/{i // (total_iter // 10)}/model.ckpt")
            saver2.save(sess, f"models/{model2.name}/{i // (total_iter // 10)}/model.ckpt")

    saver.save(sess, save_path)
    saver2.save(sess, save_path2)

def train(sess, model, env_count, num_steps, update_interval, log_interval=10, save_interval=50):
    randm = randommodel()
    
    model.make_summary()
    summaries = tf.summary.merge_all()
    writer = tf.summary.FileWriter("./logs/" + model.name, sess.graph)
    with tf.variable_scope(model.name):
        global_step = tf.Variable(initial_value=0, trainable=False, name="global_step")
        increment_global_step = tf.assign_add(global_step, update_interval*2*env_count, name = 'increment_global_step')
    save_path = "models/" + model.name + "/model.ckpt"
    sess.run(tf.global_variables_initializer())
    saver = load_model(sess, model, save_path)
    currstep = sess.run(global_step)
    runner = Runner(env_count, update_interval)
    total_iter = num_steps // (update_interval * 2 * env_count)
    curr_iter = currstep // (update_interval * 2 * env_count)
    prevtime = time.time()
    for i in range(curr_iter, total_iter+1):
        sess.run(increment_global_step)
        currstep += update_interval * 2 * env_count

        batch1, batch2 = runner.run_steps(model, model, currstep)
        
        if (i+1) % log_interval == 0:
            print(f"progress:\t{i+1}/{total_iter+1} ({round((i+1)/(total_iter+1)*100, 2)}%)")
            currtime = time.time()
            time_passed = currtime - prevtime
            print(f"elapsed time:\t{round(time_passed, 3)} second")
            print(f"time left:\t{round(time_passed*(total_iter-i)/log_interval/3600, 3)} hour")
            prevtime = currtime
            print('-----------------------------------------------------------')

        if callable(model.learning_rate):
            lr = model.learning_rate(currstep / num_steps)
        else:
            lr = model.learning_rate
        if lr <= 1e-8:
            lr = 1e-8
            batch1.extend(batch2)
        summary_data = model.train_batches(batch1, lr, summaries)
        if summary_data != None:
            writer.add_summary(summary_data, currstep)

        if (i+1) % save_interval == 0:
            saver.save(sess, save_path)
        if i % (total_iter // 10) == 0 and i != 0:
            saver.save(sess, f"models/{model.name}/{i // (total_iter // 10)}/model.ckpt")

    saver.save(sess, save_path)
    runner.close()

def run_only(sess, model, savenum=0, render=True, model2=None):
    if savenum == 0:
        save_path = "models/" + model.name + "/model.ckpt"
        if model2 != None:
            save_path2 = "models/" + model2.name + "/model.ckpt"
    else:
        save_path = f"models/{model.name}/{savenum}/model.ckpt"
        if model2 != None:
            save_path2 = f"models/{model2.name}/{savenum}/model.ckpt"
    load_model(sess, model, save_path)
    if model2 != None:
        load_model(sess, model2, save_path2)
    for i in range(100):
        if model2 == None:
            play_human(model)
        else:
            play_game(model, model2)

def trainpool(sess, model_name, env_count, num_steps, update_interval, log_interval=10, save_interval=50):
    agent_count = 3

    models = [create_model(sess, f"{model_name}-{i}") for i in range(agent_count)]
    models[0].make_summary()
    summaries = tf.summary.merge_all()
    #summaries = None
    writer = tf.summary.FileWriter("./logs/" + models[0].name, sess.graph)
    with tf.variable_scope(models[0].name):
        global_step = tf.Variable(initial_value=0, trainable=False, name="global_step")
        increment_global_step = tf.assign_add(global_step, update_interval*env_count*agent_count, name = 'increment_global_step')
    sess.run(tf.global_variables_initializer())
    
    savers = [load_model(sess, models[i], f"models/{models[i].name}/model.ckpt") for i in range(agent_count)]

    currstep = sess.run(global_step)
    runners = [Runner(env_count, update_interval) for _ in range(agent_count*agent_count)]
    total_iter = num_steps // (update_interval*env_count*agent_count)
    curr_iter = currstep // (update_interval*env_count*agent_count)
    prevtime = time.time()
    for i in range(curr_iter, total_iter+1):
        sess.run(increment_global_step)
        currstep += update_interval*env_count*agent_count
        ri = 0
        batches = [list() for _ in range(agent_count)]
        for m1 in range(agent_count):
            for m2 in range(agent_count):
                batch1, batch2 = runners[ri].run_steps(models[m1], models[m2], currstep)
                batches[m1].extend(batch1)
                batches[m2].extend(batch2)
                ri += 1

        if (i+1) % log_interval == 0:
            print(f"progress:\t{i+1}/{total_iter+1} ({round((i+1)/(total_iter+1)*100, 2)}%)")
            currtime = time.time()
            time_passed = currtime - prevtime
            print(f"elapsed time:\t{round(time_passed, 3)} second")
            print(f"time left:\t{round(time_passed*(total_iter-i)/log_interval/3600, 3)} hour")
            prevtime = currtime
            print('-----------------------------------------------------------')
        for m in range(agent_count):
            if callable(models[m].learning_rate):
                lr = models[m].learning_rate(currstep / num_steps)
            else:
                lr = models[m].learning_rate
            if lr <= 1e-8:
                lr = 1e-8
            if m == 0:
                summary_data = models[m].train_batches(batches[m], lr, summaries)
                if summary_data != None:
                    writer.add_summary(summary_data, currstep)
            else:
                models[m].train_batches(batches[m], lr)

            if (i+1) % save_interval == 0:
                savers[m].save(sess, f"models/{models[m].name}/model.ckpt")
            if i % (total_iter // 10) == 0 and i != 0:
                savers[m].save(sess, f"models/{models[m].name}/{i // (total_iter // 10)}/model.ckpt")
    for i in range(agent_count):
        savers[i].save(sess, f"models/{models[i].name}/model.ckpt")


def create_model(sess, name):
    with tf.variable_scope(name):
        input = tf.placeholder(tf.float32, [None, 12])
        initializer = tf.orthogonal_initializer(np.sqrt(2)) #Orthogonal initializer
        network = add_dense(input, 32, activation=tf.nn.tanh, kernel_initializer=initializer, name="dense1")
        network = add_dense(network, 32, activation=tf.nn.tanh, kernel_initializer=initializer, name="dense2")
        return PPO(sess, input, network, actiontype.Continuous, 2, epochs=10, minibatch_size=32, gamma=0.99, beta2=0.00, epsilon=0.2,\
            learning_rate=lambda f : 3e-4*(1-f), name=name)

def main():
    with tf.Session() as sess:
        name = 'hoc_long'
        #trainpool(sess, name, 2, 5000000, 512)

        model = create_model(sess, 'hoc_long-2')
        #model2 = create_model(sess, 'hoc_long-1')
        #train(sess, model, 24, 100000000, 256)
        #traintwo(sess, model, model2, 32, 5000000, 256)
        #run_only(sess, model, savenum=0, model2=None)
        save_path = f'models/{model.name}/1/model.ckpt'
        load_model(sess, model, save_path)
        save_session('result.txt', sess)

if __name__ == "__main__":
    main()