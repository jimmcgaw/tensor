import numpy as np
import tensorflow as tf

num_points = 1000
vectors_set = []

# generate dummy training data for a linear function
for i in xrange(num_points):
    x1 = np.random.normal(0.0, 0.55)
    # f(x) = 0.1x + 0.3 ... with noise so line isn't straight
    y1 = x1 * 0.1 + 0.3 + np.random.normal(0.0, 0.03)
    vectors_set.append([x1, y1])

x_data = [v[0] for v in vectors_set]
y_data = [v[1] for v in vectors_set]


# next goal is to derive the parameters W and b for function above, where
# f(x) = Wx + b

W = tf.Variable(tf.random_uniform([1], -1.0, 1.0))
b = tf.Variable(tf.zeros([1]))
y = W * x_data + b

# debug, see what's happening here
print tf.random_uniform([1], -1.0, 1.0)  # Tensor object
print tf.zeros([1]) # Tensor object
print W
print b
print y  # Tensor object

# error function is mean squared error
loss = tf.reduce_mean(tf.square(y-y_data))

optimizer = tf.train.GradientDescentOptimizer(0.5)
train = optimizer.minimize(loss)

init = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init)

# run gradient descent algorithm for 10 iterations
for step in xrange(10):
    sess.run(train)
    print step, sess.run(W), sess.run(b)

# (optional) graph that data with matplotlib

# import matplotlib.pyplot as plt
# # graph the data points
# plt.plot(x_data, y_data, 'ro')
# # graph our computed line of best fit from computed parameters
# plt.plot(x_data, sess.run(W) * x_data + sess.run(b))
# plt.legend()
# plt.show()
