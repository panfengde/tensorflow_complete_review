import tensorflow as tf
import numpy as np

# temp_input = np.random.normal(0, 20, (20, 2))
# w1 = np.array([
#     [1, 2],
#     [2, 3],
# ], dtype=np.float32)
# b1 = np.array([3], dtype=np.float32)
# temp_result_ = np.matmul(temp_input, w1) + b1
# temp_result = tf.constant(temp_result_, dtype=tf.float32)
#
# w2 = np.array([
#     [2],
#     [5],
# ], dtype=np.float32)
# b2 = np.array([7], dtype=np.float32)
# tep_result2 = np.matmul(temp_result_, w2) + b2
# tep_result2 = tf.constant(tep_result2, dtype=tf.float32)
# temp_input = tf.constant(temp_input, dtype=tf.float32)
temp_input = tf.constant([
    [1, 2],
    [2, 3],
    [1, 42],
    [4, 15],
    [2, 25],
    [6, 95],
    [8, 65],
    [14, 54],
    [24, 53],
    [64, 51],
    [74, 25],
], dtype=tf.float32)

temp_result = tf.constant([
    [[8, 11],
     [11, 16],
     [88, 131],
     [37, 56],
     [55, 82],
     [199, 300],
     [141, 214],
     [125, 193],
     [133, 210],
     [169, 284],
     [127, 226]]], dtype=tf.float32)

tep_result2 = tf.constant([
    [78],
    [109],
    [838],
    [361],
    [527],
    [1905],
    [1359],
    [1222],
    [1323],
    [1765],
    [1391]], dtype=tf.float32)


sess = tf.InteractiveSession()


def net_one(input, reuse=False):
    with tf.variable_scope('net_one') as scope:
        if reuse:
            scope.reuse_variables()
        w = tf.get_variable(
            name='net_w',
            shape=[2, 2],
            initializer=tf.random_normal_initializer(mean=0, stddev=1, seed=0.4),
        )
        b = tf.get_variable(name='net_b', initializer=tf.constant(1, shape=[2], dtype=tf.float32))

        result = tf.nn.relu(tf.matmul(input, w)) + b
        return result


def dc_model(input):
    with tf.variable_scope('dc_model') as scope:
        w = tf.get_variable(
            name='dc_w',
            shape=[2, 1],
            initializer=tf.random_normal_initializer(mean=0, stddev=1, seed=0.4),

        )
        b = tf.get_variable(name='dc_b', initializer=tf.ones([1], dtype=tf.float32))
        result = tf.nn.relu(tf.matmul(input, w)) + b
        return result


net_one_result = net_one(temp_input)
net_one_loss = tf.reduce_mean(tf.square(net_one_result - temp_result))
tf.summary.scalar('net_one_loss', net_one_loss)
op_Adam_net_one = tf.train.GradientDescentOptimizer(0.01, name='net_one_op').minimize(net_one_loss)

dc_model_input = net_one(temp_input, reuse=True)
dc_model_result = dc_model(dc_model_input)
dc_model_loss = tf.reduce_mean(tf.square(dc_model_result - tep_result2))
tf.summary.scalar('dc_model_loss', dc_model_loss)
t_vars = tf.trainable_variables()
need_var = [var for var in t_vars if 'dc_' in var.name]
op_Adam_dc_model = tf.train.GradientDescentOptimizer(0.01, name='dc_model_op').minimize(dc_model_loss,
                                                                                        var_list=need_var)

writer = tf.summary.FileWriter("F:\\python历程\\tensorflow完整学习笔记\\tensorflow_complete_review\\5variable\\data",
                               sess.graph)
summaries = tf.summary.merge_all()
sess.run(tf.global_variables_initializer())
for i in range(2000):
    sess.run(op_Adam_net_one)
    if i % 100 == 0:
        summ1 = sess.run(summaries)
        writer.add_summary(summ1, global_step=i)

with tf.variable_scope('net_one', reuse=True):
    a = tf.get_variable('net_w')
    b = tf.get_variable('net_b')
    print(a.eval())
    print(b.eval())

with tf.variable_scope('dc_model', reuse=True):
    a = tf.get_variable('dc_w')
    b = tf.get_variable('dc_b')
    print(a.eval())
    print(b.eval())

for i in range(2000):
    sess.run(op_Adam_dc_model)
    if i % 100 == 0:
        summ2 = sess.run(summaries)
        writer.add_summary(summ2, global_step=i + 1999)

with tf.variable_scope('net_one', reuse=True):
    a = tf.get_variable('net_w')
    b = tf.get_variable('net_b')
    print('--------------------')
    print(a.eval())
    print(b.eval())
with tf.variable_scope('dc_model', reuse=True):
    a = tf.get_variable('dc_w')
    b = tf.get_variable('dc_b')
    print(a.eval())
    print(b.eval())
