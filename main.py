import tensorflow as tf
import numpy as np

sess = tf.Session()


def one():
    node1 = tf.constant(3.0, tf.float32)
    node2 = tf.constant(4.0)
    print node1, node2
    print sess.run([node1, node2])

    node3 = tf.add(node1, node2)
    print 'node3', node3
    print sess.run(node3)


def two():
    a = tf.placeholder(tf.float32)
    b = tf.placeholder(tf.float32)
    adder_node= a + b

    print sess.run(adder_node, {a: 3, b: 4.5})
    print sess.run(adder_node, {a: 3, b: [4, 5]})

    add_and_triple = adder_node * 3.
    print sess.run(add_and_triple, {a: 3, b: 4.5})


def three():
    W = tf.Variable([.3], tf.float32)
    b = tf.Variable([-.3], tf.float32)
    x = tf.placeholder(tf.float32)
    linear_model = W * x + b

    init = tf.global_variables_initializer()
    sess.run(init)

    x_train = [1,2,3,4]
    y_train = [0,-1,-2,-3]

    print sess.run(linear_model, {x: x_train})

    y = tf.placeholder(tf.float32)
    squared_deltas = tf.square(linear_model - y)
    loss = tf.reduce_sum(squared_deltas)
    print sess.run(loss, {x:x_train, y:y_train})

    # fixW = tf.assign(W, [-1.])
    # fixb = tf.assign(b, [1.])
    # sess.run([fixW, fixb])
    # print sess.run(loss, {x:x_train, y:y_train})

    optimizer = tf.train.GradientDescentOptimizer(0.01)
    train = optimizer.minimize(loss)

    for i in range(1000):
        sess.run(train, {x:x_train, y:y_train})

    curr_W, curr_b, curr_loss = sess.run([W, b, loss], {x:x_train, y:y_train})
    print 'W: %s, b: %s, loss: %s' % (curr_W, curr_b, curr_loss)


def four():
    features = [tf.contrib.layers.real_valued_column('x', dimension=1)]
    estimator = tf.contrib.learn.LinearRegressor(feature_columns=features)
    x = np.array([1.,2.,3.,4.])
    y = np.array([0., -1., -2., -3.])
    input_fn = tf.contrib.learn.io.numpy_input_fn(
        {'x': x}, y, batch_size=4, num_epochs=1000)
    estimator.fit(input_fn=input_fn, steps=1000)
    print estimator.evaluate(input_fn=input_fn)


if __name__ == '__main__':
    four()
