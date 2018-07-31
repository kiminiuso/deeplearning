import tensorflow as tf
import numpy
import random
from sklearn.cross_validation import train_test_split

seed = 7
numpy.random.seed(seed)

batch_size = 16

#加载数据
dataset = numpy.loadtxt("pima-indians-diabetes.csv", delimiter=",")
# 区分前8个输入和第九个输出
X = dataset[0:len(dataset), 0:8]
Y_ = dataset[0:len(dataset), 8]
Y = []

for i in Y_[0:len(dataset)]:
    batch = [i]
    Y.append(batch)

# 自动分组，好像没卵用，样本太少
# X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.33, random_state=0)

# 定义节点，接收数据
x = tf.placeholder(tf.float32, shape=[None, 8], name='x-input')
y_ = tf.placeholder(tf.float32, shape=[None, 1], name='y-input')

dense1 = tf.layers.dense(inputs=x, units=12, activation=tf.nn.relu)
dense2 = tf.layers.dense(inputs=dense1, units=8, activation=tf.nn.sigmoid)
y = tf.layers.dense(inputs=dense2, units=1, activation=None)

# 4.定义 loss 表达式
# the error between dense3 and real data
loss = tf.reduce_mean(tf.reduce_sum(tf.square(y_ - y),reduction_indices=[1]))


# 5.选择 optimizer 使 loss 达到最小,要用adam优化
train_step = tf.train.AdamOptimizer(learning_rate=0.01).minimize(loss)

with tf.Session() as sess:

    # 对所有变量进行初始化
    init = tf.global_variables_initializer()
    # 上面定义的都没有运算，直到 sess.run 才会开始运算
    sess.run(init)

    # 迭代 1000 次学习，sess.run optimizer
    for i in range(1000):
        start = random.randint(0, len(dataset) - batch_size)
        end = start + batch_size
        # training train_step 和 loss 都是由 placeholder 定义的运算，所以这里要用 feed 传入参数
        sess.run(train_step, feed_dict={x: X[start:end], y_: Y[start:end]})
        if i % 5 == 0:
           # 计算精确度、损失率
           correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
           accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
           print("第 %g 次训练，损失率 %g ， 正确率 %g" % (i,
                                               sess.run(loss, feed_dict={x: X, y_: Y}),
                                               sess.run(accuracy, feed_dict={x: X, y_: Y})))
