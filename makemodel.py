import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

## This is where the data gets imported
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

# Model Creation - Placeholders & Variables
x = tf.placeholder(tf.float32, [None, 784])
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))
y = tf.nn.softmax(tf.matmul(x, W) + b)

# Defining the loss and optimizations
y_ = tf.placeholder(tf.float32, [None, 10])
cross_entropy = -tf.reduce_sum(y_ * tf.log(y))
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

init = tf.global_variables_initializer()
saver = tf.train.Saver()

# The model is saved to disk as a model.ckpt file

# Initializing the Session
with tf.Session() as sess:
    sess.run(init)
    for i in range(1000):
        batch_xs, batch_ys = mnist.train.next_batch(100)
        sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

    print (sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))
    save_path = saver.save(sess, "./model/")
print ("Model saved in file: ", save_path)

