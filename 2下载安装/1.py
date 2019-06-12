import tensorflow as tf

hello = tf.constant('hello,Tensorflow')
sess = tf.Session()
print(sess.run(hello))

a = tf.constant(10)
b = tf.constant(20)
print(sess.run(a + b))

aa = tf.constant([1, 2, 3, 4])
bb = tf.constant([12, 22, 32, 42])
print(sess.run(aa * bb))


sess.close()
