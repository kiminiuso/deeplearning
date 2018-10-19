import tensorflow as tf
from PIL import Image

tfrecords_filename = "data.tfrecords"

filename_queue = tf.train.string_input_producer([tfrecords_filename], )  # 读入流中
reader = tf.TFRecordReader()
_, serialized_example = reader.read(filename_queue)  # 返回文件名和文件
features = tf.parse_single_example(serialized_example,
                                   features={
                                       'label': tf.FixedLenFeature([], tf.string),
                                       'img_raw': tf.FixedLenFeature([], tf.string),
                                   })  # 取出包含image和label的feature对象
image = tf.decode_raw(features['img_raw'], tf.uint8)
image = tf.reshape(image, [32, -1, 3])
label = tf.cast(features['label'], tf.string)

with tf.Session() as sess:  # 开始一个会话
    init_op = tf.global_variables_initializer()
    sess.run(init_op)
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)
    for i in range(100, 102):
        example, l = sess.run([image, label])  # 在会话中取出image和label
        print(l)
        # example = tf.reshape(example, [32, len(example)/96 , 3])
        img = Image.fromarray(example, 'RGB')  # 这里Image是之前提到的
        img.show()

    coord.request_stop()
    coord.join(threads)
