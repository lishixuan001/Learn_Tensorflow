import tensorflow as tf

# Variables in tensorflow must be defined
# Defining a variable can give it an init value and a name
# Or you can leave it blank for just declaring
state = tf.Variable(0, name="counter")
# print(state.name)

one = tf.constant(1)

new_value = tf.add(state, one)
# Assigning an update method so that when next run, it updates as declared
update = tf.assign(state, new_value)

# 如果有定义variable，一定要记得定义init
init = tf.global_variables_initializer() # Must have if define variable

with tf.Session() as sess:
    sess.run(init)
    for _ in range(3):
        sess.run(update)
        print(sess.run(state)) # Note that must run the state to make it print the value
