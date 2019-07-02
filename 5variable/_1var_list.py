import tensorflow as tf

const_1 = tf.constant([
    [1, 2],
    [2, 3],
    [4, 2],
    [3, 1],
], dtype=tf.float32)

result = tf.constant([
    [1],
    [1],
    [1],
    [1],
], dtype=tf.float32)

with tf.variable_scope('scope_1') as scope:
    var_1 = tf.get_variable('var_1', [2, 2], initializer=tf.random_normal_initializer(mean=0, stddev=1, seed=0.4))
    var_2 = tf.get_variable('he_2', [2, 1], initializer=tf.random_normal_initializer(mean=0, stddev=1, seed=0.3))
    var_b = tf.get_variable('var_b', initializer=tf.zeros([1], dtype=tf.float32))
    var_x = tf.get_variable('var_x', initializer=tf.zeros([1], dtype=tf.float32), trainable=False)

t_vars = tf.trainable_variables()
print(t_vars)
need_var = [var for var in t_vars if 'var_' in var.name]

loss = tf.reduce_mean(tf.square(tf.nn.relu(tf.matmul(tf.matmul(const_1, var_1), var_2)) + var_b - result))
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print(sess.run(var_1))
    print(sess.run(var_2))
    print(sess.run(var_b))
    op_Adam = tf.train.GradientDescentOptimizer(0.5).minimize(loss, var_list=need_var)
    for i in range(100):
        sess.run(op_Adam)
        if (i % 10) == 0:
            print('----------------', sess.run(loss))
    print('结束')
    print(sess.run(var_1))
    print(sess.run(var_2))
    print(sess.run(var_b))
