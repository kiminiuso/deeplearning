import tensorflow as tf
import numpy as np
from numpy.random import RandomState
from sklearn.cross_validation import train_test_split
from sklearn import preprocessing

#1维输出、计算准确率要确定一个阈值，不知道怎么确定

batch_size = 8
x = tf.placeholder(tf.float32, shape=(None, 8), name='x-input')
y_= tf.placeholder(tf.float32, shape=(None, 1), name='y-input')

w1 = tf.Variable(tf.random_normal([8, 12], stddev=1, seed=1))
w2 = tf.Variable(tf.random_normal([12, 8], stddev=1, seed=1))
w3 = tf.Variable(tf.random_normal([8, 1], stddev=1, seed=1))

bias1 = tf.Variable(tf.random_uniform([12]))
bias2 = tf.Variable(tf.random_uniform([8]))
bias3 = tf.Variable(tf.random_uniform([1]))

a = tf.matmul(x, w1) + bias1
a = tf.nn.relu(a)
b = tf.matmul(a, w2) + bias2
b = tf.nn.relu(b)
y = tf.matmul(b, w3) + bias3
y = tf.nn.sigmoid(y)

#loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=y, labels=y_))
#loss = -tf.reduce_mean(y_ * tf.log(tf.clip_by_value(y, 1e-10, 1.0)))
loss = tf.reduce_mean(tf.reduce_sum(tf.square(y_-y), reduction_indices=[1]))
train_step = tf.train.AdamOptimizer(0.001).minimize(loss)

dataset = np.loadtxt("pima-indians-diabetes.csv", delimiter=",")
X = dataset[:,0:8]
min_max_scaler = preprocessing.MinMaxScaler()
X = min_max_scaler.fit_transform(X)
Y = dataset[:,8]
Y = np.array(Y).reshape((-1,1))
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.33, random_state=0)
print('###')
print(X_train)
print(Y_train)
print(X_test)
print(Y_test)

with tf.Session() as sess:
    init_op = tf.global_variables_initializer()
    sess.run(init_op)

    epochs = 500
    for e in range(epochs):
        total_batch = int(len(X_train) / batch_size)
        X_batches = np.array_split(X_train, total_batch)
        Y_batches = np.array_split(Y_train, total_batch)
        for i in range(total_batch):
            sess.run(train_step, feed_dict={x: X_batches[i], y_: Y_batches[i]})

        if e % 10 == 0:
            total_loss = sess.run(loss, feed_dict={x: X, y_: Y})
            print("After %d training step(s), cross entropy on all data is %g" % (e, total_loss))

    print('y_', Y_train[0:10])
    print('y', sess.run(y, feed_dict={x: X_train[0:10]}))
