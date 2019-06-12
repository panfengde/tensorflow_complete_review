import tensorflow as tf
import input_data

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

x = tf.placeholder(tf.float32, [None, 784])
y_ = tf.placeholder(tf.float32, [None, 10])

# w = tf.Variable(tf.zeros([784, 10]))
# b = tf.Variable(tf.zeros([10]))

w = tf.Variable(tf.random_uniform([784, 10], -1.0, 1.0))
b = tf.Variable(tf.random_uniform([10], -1.0, 1.0))
y = tf.nn.softmax(tf.matmul(x, w) + b)
cross_entropy = -tf.reduce_sum(y_ * tf.log(y))

train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)
init = tf.initialize_all_variables()
with tf.Session() as sess:
    sess.run(init)
    for i in range(1000):
        batch_xs, batch_ys = mnist.train.next_batch(100)
        sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, 'float'))
    the_accuracy, pre = sess.run([accuracy,correct_prediction],
                                     feed_dict={x: mnist.test.images, y_: mnist.test.labels})
    bb, ww = sess.run([b, w])
    print(the_accuracy)
    print(pre[0:100])

