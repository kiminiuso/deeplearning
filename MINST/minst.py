import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from minst_model1  import mymodel

mnist = input_data.read_data_sets("./MINST_data/", one_hot=True)

# 创建两个占位符，x为输入网络的图像，y_为输入网络的图像类别
x = tf.placeholder("float", shape=[None, 784])
y_ = tf.placeholder("float", shape=[None, 10])


# 权重初始化函数
def weight_variable(shape):
    # 输出服从截尾正态分布的随机值
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


# 偏置初始化函数
def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


y_conv,keep_prob = mymodel(x)
# 预测值和真实值之间的交叉墒
cross_entropy = -tf.reduce_sum(y_ * tf.log(y_conv))

# train op, 使用ADAM优化器来做梯度下降。学习率为0.0001
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

# 评估模型，tf.argmax能给出某个tensor对象在某一维上数据最大值的索引。
# 因为标签是由0,1组成了one-hot vector，返回的索引就是数值为1的位置
correct_predict = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))

# 计算正确预测项的比例，因为tf.equal返回的是布尔值，
# 使用tf.cast把布尔值转换成浮点数，然后用tf.reduce_mean求平均值
accuracy = tf.reduce_mean(tf.cast(correct_predict, "float"))

# saver = tf.train.Saver()

# 开始训练模型，循环20000次，每次随机从训练集中抓取50幅图像
with tf.Session() as sess:
    # 创建一个交互式Session
    sess.run(tf.global_variables_initializer())
    for i in range(20000):
        batch = mnist.train.next_batch(50)
        if i % 100 == 0:
            # 每100次输出一次日志
            train_accuracy = accuracy.eval(feed_dict={
                x: batch[0], y_: batch[1], keep_prob: 1.0})
            print("step %d, training accuracy %g" % (i, train_accuracy))
            # saver.save(sess, 'model')
        train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})

    # 预测
    # saver = tf.train.Saver(tf.global_variables())
    # saver.restore(sess, 'model')
    print("test accuracy %g" % accuracy.eval(feed_dict={
        x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))
