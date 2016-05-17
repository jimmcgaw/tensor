import tensorflow as tf

# declare 2 symbolic variables
a = tf.placeholder("float")
b = tf.placeholder("float")

y = tf.mul(a, b)

# create a session, this is where we connect to TensorFlow's libraries.
sess = tf.Session()

print sess.run(y, feed_dict={a: 3, b: 3 })
