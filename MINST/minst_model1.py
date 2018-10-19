import tensorflow as tf

# 权重初始化函数
def weight_variable(shape):
    # 输出服从截尾正态分布的随机值
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


# 偏置初始化函数
def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


# 创建卷积op
# x 是一个4维张量，shape为[batch,height,width,channels]
# 卷积核移动步长为1。填充类型为SAME,可以不丢弃任何像素点, VALID丢弃边缘像素点
# 感受野
# def conv2d(x, W, padding):
#     return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding=padding)


# 创建池化op
# 采用最大池化，也就是取窗口中的最大值作为结果
# x 是一个4维张量，shape为[batch,height,width,channels]
# ksize表示pool窗口大小为2x2,也就是高2，宽2
# strides，表示在height和width维度上的步长都为2
def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1], padding="SAME")


def mymodel(input_x):
    # 第1层，卷积层
    # 初始化W为[5,5,1,6]的张量，表示卷积核大小为5*5，3表示图像通道数，6表示卷积核个数即输出6个特征图
    W_conv1 = weight_variable([5, 5, 1, 6])
    # 初始化b为[6],即输出大小
    b_conv1 = bias_variable([6])

    # 把输入x(二维张量,shape为[batch, 784])变成4d的x_image，x_image的shape应该是[batch,28,28,1]
    # -1表示自动推测这个维度的size
    x_image = tf.reshape(input_x, [-1, 28, 28, 1])

    # 把x_image和权重进行卷积，加上偏置项，然后应用ReLU激活函数，最后进行max_pooling
    # h_pool1的输出即为第一层网络输出，shape为[batch,14,14,6]
    h_conv1 = tf.nn.relu(tf.nn.conv2d(x_image, W_conv1, strides=[1, 1, 1, 1], padding='SAME') + b_conv1)
    h_pool1 = max_pool_2x2(h_conv1)

    # 第2层，卷积层
    # 卷积核大小依然是5*5，通道数为6，卷积核个数为16
    W_conv2 = weight_variable([5, 5, 6, 16])
    b_conv2 = weight_variable([16])

    # h_pool2即为第二层网络输出，shape为[batch,5,5,16]
    h_conv2 = tf.nn.relu(tf.nn.conv2d(h_pool1, W_conv2, strides=[1, 1, 1, 1], padding='VALID') + b_conv2)
    h_pool2 = max_pool_2x2(h_conv2)

    # 第3层, 全连接层
    # 这层是拥有120个神经元的全连接层
    # W的第1维size为5*5*16，5*5是h_pool2输出的size，16是第2层输出神经元个数
    W_fc1 = weight_variable([5 * 5 * 16, 120])
    b_fc1 = bias_variable([120])

    # 计算前需要把第2层的输出reshape成[batch, 5*5*16]的张量
    h_pool2_flat = tf.reshape(h_pool2, [-1, 5 * 5 * 16])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

    # 第四层全连接层
    # 120个神经元转换成100个神经元，用tanh回归
    W_fc4 = weight_variable([120, 100])
    b_fc4 = bias_variable([100])

    h_pool4_flat = tf.reshape(h_fc1, [-1, 120])
    h_fc4 = tf.nn.tanh(tf.matmul(h_pool4_flat, W_fc4) + b_fc4)

    # Dropout层
    # 为了减少过拟合，在输出层前加入dropout
    keep_prob = tf.placeholder("float")
    h_fc1_drop = tf.nn.dropout(h_fc4, keep_prob)

    # 输出层
    # 最后，添加一个softmax层
    # 可以理解为另一个全连接层，只不过输出时使用softmax将网络输出值转换成了概率
    W_out = weight_variable([100, 10])
    b_out = bias_variable([10])

    y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, W_out) + b_out)

    return y_conv,keep_prob
