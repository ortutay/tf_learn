import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('MNIST_data/', one_hot=True)

# x is the input data, 784 pixels flattened. Use 'None' to indicate
# that number of examples is variable.
x = tf.placeholder(tf.float32, [None, 784])

# W is the weights matrix and b is the bias. We will train these as
# part of our classifier.
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

# y is the result from our classifier.
y = tf.nn.softmax(tf.matmul(x, W) + b)

# y_ is the true data.
y_ = tf.placeholder(tf.float32, [None, 10])

# cross_entropy represents the loss.
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))

train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

sess = tf.InteractiveSession()
tf.global_variables_initializer().run()

for _ in range(1000):
    batch_x, batch_y = mnist.train.next_batch(100)
    sess.run(train_step, feed_dict={x: batch_x, y_: batch_y})

correct_predictions = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
print 'correct_predictions', correct_predictions
accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32))
print sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels})
