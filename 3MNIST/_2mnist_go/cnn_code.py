import tensorflow as tf

import sys
import os

current_dir = os.path.abspath(os.path.dirname(__file__))
sys.path.append(current_dir + '/..')
import mnist_data.input_data as input_data

mnist = input_data.read_data_sets(current_dir + "/../mnist_data/MNIST_data/", one_hot=True)

sess = tf.InteractiveSession()

# 占位符
x = tf.placeholder("float", shape=[None, 784])
y_ = tf.placeholder("float", shape=[None, 10])


# 权重初始化函数
def weight_variable(shape):
    initial = tf.truncated_normal(shape=shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


# 卷积池化函数
def conv2d(x, w):
    return tf.nn.conv2d(x, w, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


# 第一层卷积
# variable_scope
with tf.name_scope('layer1'):
    w_conv1 = weight_variable([5, 5, 1, 32])  # ---32个5*5*!的卷积核心
    b_conv1 = bias_variable([32])
    x_image = tf.reshape(x, [-1, 28, 28, 1])
    h_conv1 = tf.nn.relu(conv2d(x_image, w_conv1) + b_conv1)
    h_pool1 = max_pool_2x2(h_conv1)

# 第二层卷积
with tf.name_scope('layer2'):
    w_conv2 = weight_variable([5, 5, 32, 64])  # ---64个5*5*32的卷积核心
    b_conv2 = bias_variable([64])
    h_conv2 = tf.nn.relu(conv2d(h_pool1, w_conv2) + b_conv2)
    h_pool2 = max_pool_2x2(h_conv2)

# 全连接层
with tf.name_scope('layer3'):
    w_fc1 = weight_variable(([7 * 7 * 64, 1024]))
    b_fc1 = bias_variable([1024])
    h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, w_fc1) + b_fc1)

# drop_out层
with tf.name_scope('layer_drop_out'):
    keep_prob = tf.placeholder("float")
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

# 输出层
with tf.name_scope('layer_out'):
    w_fc2 = weight_variable([1024, 10])
    b_fc2 = bias_variable([10])
    y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, w_fc2) + b_fc2, name='the_result')

# 训练和评估模型
with tf.name_scope('Assessment'):
    cross_entropy = -tf.reduce_mean(y_ * tf.log(y_conv))
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
    correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, 'float'))
    tf.summary.scalar('cross_entropy', cross_entropy)
    tf.summary.scalar('accuracy', accuracy)

# Use `tf.global_variables_initializer` instead.

print('----')
print(current_dir + "//..//..//tensorboard//test")

writer = tf.summary.FileWriter(current_dir + "//..//..//tensorboard//test",
                               sess.graph)

summaries = tf.summary.merge_all()
saver = tf.train.Saver(max_to_keep=3)
sess.run(tf.global_variables_initializer())
model_file = tf.train.latest_checkpoint('./save')

if model_file:
    saver.restore(sess, model_file)
for i in range(10000):
    batch = mnist.train.next_batch(50)
    if i % 100 == 0:
        summ = sess.run(summaries, feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.7})
        writer.add_summary(summ, global_step=i)

    else:
        train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.7})
    if (i + 1) % 1000 == 0:
        saver.save(sess, "./save/train_epoch_" + str(i) + ".ckpt")

print("test accuracy %g" % accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))
# localhost:6006
# tensorboard --logdir=F:\python历程\tensorflow完整学习笔记\tensorflow_complete_review\tensorboard\test
