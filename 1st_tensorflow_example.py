import tensorflow as tf
import numpy as np

# Create data
x_data = np.random.rand(100).astype(np.float32)
y_data = x_data * 0.2 + 0.3

### Create tensorflow structure start ###
''' 规定参数变量 '''
Weights = tf.Variable(tf.random_uniform([1], -1.0, 1.0)) # Use capital "Weights since it is possible that it is a matrix"
biases = tf.Variable(tf.zeros([1]))

''' 设置预测的y，要提升y的准确度 '''
y = Weights * x_data + biases

''' Optimize 预测值和真实的y的误差。用一个优化器减少误差 '''
loss = tf.reduce_mean(tf.square(y - y_data))
optimizer = tf.train.GradientDescentOptimizer(0.5) # "0.5"是指随即效率
train = optimizer.minimize(loss)

''' 初始化变量 '''
init = tf.global_variables_initializer()
### Create tensorflow structure ened ###

# 定义一个"Sesson"
sess = tf.Session()
sess.run(init) # 像一个指针，指向要处理的地方，那个地方就被激活了

for step in range(201):
    sess.run(train)
    if step % 20 == 0:
        print(step, sess.run(Weights), sess.run(biases))
