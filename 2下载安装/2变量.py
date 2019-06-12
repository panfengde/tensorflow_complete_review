import tensorflow as tf
#构建图
state=tf.Variable(0,name='counter')
one=tf.constant(1)
new_value=tf.add(state,one)
update=tf.assign(state,new_value)

init_op=tf.initialize_all_variables()


#启动图
with tf.Session() as sess:
    sess.run(init_op)
    for x in range(3):
        sess.run(update)
        print(sess.run(state))
        print(state)