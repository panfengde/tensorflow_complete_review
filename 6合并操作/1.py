import tensorflow as tf
import numpy as np

# 构建图
t1 = [[1, 2, 3], [4, 5, 6]]
t2 = [[7, 8, 9], [10, 11, 12]]
# tf.concat(0, [t1, t2])
# tf.concat(1, [t1, t2])
x = np.array([[2, 3, 4]])
one = tf.constant([[1, 2, 3], [1, 2, 3]])
r = t1 * x
# 启动图
with tf.Session() as sess:
    # print(sess.run(tf.concat([t1, t2], 1)))
    print(sess.run(r))
