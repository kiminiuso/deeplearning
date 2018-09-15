import tensorflow as tf
import numpy as np
from numpy.random import RandomState
from sklearn.cross_validation import train_test_split
from sklearn import preprocessing

#输出改为2维，通过比较概率来计算准确率

dataset = np.loadtxt("pima-indians-diabetes.csv", delimiter=",")
X = dataset[:, 0:8]
# 归一化，没有这一步回归不了。。。想不通
min_max_scaler = preprocessing.MinMaxScaler()
X = min_max_scaler.fit_transform(X)
Y_ = dataset[:, 8]
Y = []
# Y = dataset[:, 8]
# Y = np.array(Y).reshape((-1, 1))
#[没生病，生病]
for yyyy in Y_:
    if yyyy == 0:
        Y.append([1, 0])
    else:
        Y.append([0, 1])

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.33, random_state=0)
# print(Y_test)

batch_size = 8

x = tf.placeholder(tf.float32, shape=(None, 8), name='x-input')
y_ = tf.placeholder(tf.float32, shape=(None, 2), name='y-input')

w1 = tf.Variable(tf.random_normal([8, 12], stddev=1, seed=1))
w2 = tf.Variable(tf.random_normal([12, 16], stddev=1, seed=1))
w3 = tf.Variable(tf.random_normal([16, 8], stddev=1, seed=1))
w4 = tf.Variable(tf.random_normal([8, 2], stddev=1, seed=1))

bias1 = tf.Variable(tf.random_uniform([12]))
bias2 = tf.Variable(tf.random_uniform([16]))
bias3 = tf.Variable(tf.random_uniform([8]))
bias4 = tf.Variable(tf.random_uniform([2]))

a = tf.matmul(x, w1) + bias1
a = tf.nn.relu(a)
b = tf.matmul(a, w2) + bias2
b = tf.nn.relu(b)
c = tf.matmul(b, w3) + bias3
c = tf.nn.relu(c)
y = tf.matmul(c, w4) + bias4
y = tf.nn.sigmoid(y)

loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=y, labels=y_))
# loss = -tf.reduce_mean(y_ * tf.log(tf.clip_by_value(y, 1e-10, 1.0)))
# loss = tf.reduce_mean(tf.reduce_sum(tf.square(y_-y), reduction_indices=[1]))
train_step = tf.train.AdamOptimizer(0.01).minimize(loss)


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
            # total_loss = sess.run(loss, feed_dict={x: X, y_: Y})
            # print("After %d training step(s), cross entropy on all data is %g" % (e, total_loss))
            correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
            print("After %d training step(s) : 损失率 %g ， 在测试集上的正确率 %g， 在全集上的正确率 %g" % (e, sess.run(loss, feed_dict={x: X_train, y_: Y_train}),
                                       sess.run(accuracy, feed_dict={x: X_test, y_: Y_test}),sess.run(accuracy, feed_dict={x: X, y_: Y})))

    print('y_ : ', Y_test[0:9])
    print('y : ', sess.run(y, feed_dict={x: X_test[0:9]}))



