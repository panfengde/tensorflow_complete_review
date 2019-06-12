import tensorflow as tf

a = [1, 2, 3, 4]
b = [1, 3, 4, 5]
arg = [1, 2, 3, 4, 5, 6, 8]
arg2 = [
    [1, 12, 3, 4, 5, 6, 8],
    [1, 2, 13, 4, 5, 6, 8],
    [1, 2, 3, 4, 15, 6, 8],
]
result = tf.equal(a, b)

bul = [True, False, False, False]

with tf.Session() as sess:
    get_result = sess.run(result)
    print(get_result)
    print(sess.run(tf.argmax(arg)))
    print(sess.run(tf.argmax(arg2, 1)))
    print(sess.run(tf.cast(bul, 'float')))
