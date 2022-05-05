import tensorflow as tf
import numpy as np


def get_weight(shape, regularizer):
    weights = tf.get_variable(name="weights", shape=shape,
                              initializer=tf.truncated_normal_initializer(stddev=1))
    if regularizer != None:
        tf.add_to_collection('losses', regularizer(weights))
        # print('regularizer(weight).shape=', np.shape(regularizer(weights)))
    return weights


# 11.3小结内容
def variable_summares(var, name, image):
    with tf.name_scope('summaries'):
        tf.summary.histogram(name, var)
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean/' + name, mean)
        stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar('stddev/' + name, stddev)
    if image != None:
        with tf.name_scope('IMAGE'):
            image_shaped = tf.reshape(var, [-1, var.shape[1], var.shape[2], 1])
            tf.summary.image(name + '/out', image_shaped, 3)


def inference(images, batch_size, n_clasess, regularizer):
    with tf.variable_scope('conv1') as scope:
        conv1_w = get_weight([7, 7, 3, 64], regularizer)
        conv1_b = tf.Variable(tf.constant(value=0.1, dtype=tf.float32, shape=[64]),
                              name='biases', dtype=tf.float32)
        conv = tf.nn.conv2d(images, conv1_w, strides=[1, 2, 2, 1], padding='VALID')
        pre_activation = tf.nn.bias_add(conv, conv1_b)
        conv1 = tf.nn.relu(pre_activation, name=scope.name)  # name: A name for the operation (optional).

        #variable_summares(images, 'in_image', 1)
        variable_summares(conv1_w, 'conv1/W', None)
        variable_summares(conv1_b, 'conv1/b', None)
        variable_summares(conv1, 'conv1/out', 1)
        with tf.variable_scope('vsion'):
            '''x_min = tf.reduce_min(conv1_w)
            x_max = tf.reduce_max(conv1_w)
            w_0_to_1 = (conv1_w - x_min) / (x_max - x_min)
            w_transposed = tf.transpose(w_0_to_1, [3, 0, 1, 2])
            tf.summary.image('filters', w_transposed, max_outputs=3)'''
            layer1_image1 = conv1[0:1, :, :, 0:16]
            layer1_image1 = tf.transpose(layer1_image1, perm=[3, 1, 2, 0])
            tf.summary.image("filtered_images_layer1", layer1_image1, max_outputs=5)

    with tf.variable_scope('Max_Pooling1') as scope:
        pool1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID',
                               name='pooling1')
        # 池化后进行lrn()局部响应归一化，对训练有利。咱也不太懂？？？---白面深度学习上有
        norm1 = tf.nn.lrn(pool1, depth_radius=4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name='norm1')
        with tf.variable_scope('vsion'):
            layer1_image1 = norm1[0:1, :, :, 0:16]
            layer1_image1 = tf.transpose(layer1_image1, perm=[3, 1, 2, 0])
            tf.summary.image("filtered_images_layer1", layer1_image1, max_outputs=5)

    with tf.variable_scope('conv2') as scope:
        conv2_w = get_weight([3, 3, 64, 128], regularizer)
        # conv2_w = tf.Variable(tf.truncated_normal(shape=[3, 3, 64, 128], stddev=1.0, dtype=tf.float32),
        #                      name='weights', dtype=tf.float32)
        conv2_b = tf.Variable(tf.constant(value=0.1, dtype=tf.float32, shape=[128]),
                              name='biases', dtype=tf.float32)
        conv = tf.nn.conv2d(norm1, conv2_w, strides=[1, 1, 1, 1], padding='SAME')
        pre_activation = tf.nn.bias_add(conv, conv2_b)
        conv2 = tf.nn.relu(pre_activation, name=scope.name)

        variable_summares(conv2_w, 'conv2/W', None)
        variable_summares(conv2_b, 'conv2/b', None)
        variable_summares(conv2, 'conv2/out', 1)
        with tf.variable_scope('vsion'):
            layer1_image1 = conv2[0:1, :, :, 0:16]
            layer1_image1 = tf.transpose(layer1_image1, perm=[3, 1, 2, 0])
            tf.summary.image("filtered_images_layer1", layer1_image1, max_outputs=5)

    with tf.variable_scope('Max_Pooling2') as scope:
        pool2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID',
                               name='pooling2')
        norm2 = tf.nn.lrn(pool2, depth_radius=4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name='norm2')
        with tf.variable_scope('vsion'):
            layer1_image1 = norm2[0:1, :, :, 0:16]
            layer1_image1 = tf.transpose(layer1_image1, perm=[3, 1, 2, 0])
            tf.summary.image("filtered_images_layer1", layer1_image1, max_outputs=5)

    with tf.variable_scope('conv3') as scope:
        conv3_w = get_weight([3, 3, 128, 256], regularizer)
        # conv3_w = tf.Variable(tf.truncated_normal(shape=[3, 3, 128, 256], stddev=1.0, dtype=tf.float32),
        #                      name='weights', dtype=tf.float32)
        conv3_b = tf.Variable(tf.constant(value=0.1, dtype=tf.float32, shape=[256]),
                              name='biases', dtype=tf.float32)
        conv = tf.nn.conv2d(norm2, conv3_w, strides=[1, 1, 1, 1], padding='SAME')
        pre_activation = tf.nn.bias_add(conv, conv3_b)
        conv3 = tf.nn.relu(pre_activation, name=scope.name)

        variable_summares(conv3_w, 'conv3/W', None)
        variable_summares(conv3_b, 'conv3/b', None)
        variable_summares(conv3, 'conv3/out', 1)
        with tf.variable_scope('vsion'):
            layer1_image1 = conv3[0:1, :, :, 0:16]
            layer1_image1 = tf.transpose(layer1_image1, perm=[3, 1, 2, 0])
            tf.summary.image("filtered_images_layer1", layer1_image1, max_outputs=5)

    with tf.variable_scope('Max_Pooling3') as scope:
        pool3 = tf.nn.max_pool(conv3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID',
                               name='pooling3')
        norm3 = tf.nn.lrn(pool3, depth_radius=4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name='norm3')
        with tf.variable_scope('vsion'):
            layer1_image1 = norm3[0:1, :, :, 0:16]
            layer1_image1 = tf.transpose(layer1_image1, perm=[3, 1, 2, 0])
            tf.summary.image("filtered_images_layer1", layer1_image1, max_outputs=5)

    with tf.variable_scope('conv4') as scope:
        conv4_w = get_weight([3, 3, 256, 512], regularizer)
        # conv4_w = tf.Variable(tf.truncated_normal(shape=[3, 3, 256, 512], stddev=1.0, dtype=tf.float32),
        #                      name='weights', dtype=tf.float32)
        conv4_b = tf.Variable(tf.constant(value=0.1, dtype=tf.float32, shape=[512]),
                              name='biases', dtype=tf.float32)
        conv = tf.nn.conv2d(norm3, conv4_w, strides=[1, 1, 1, 1], padding='SAME')
        pre_activation = tf.nn.bias_add(conv, conv4_b)
        conv4 = tf.nn.relu(pre_activation, name=scope.name)

        variable_summares(conv4_w, 'conv4/W', None)
        variable_summares(conv4_b, 'conv4/b', None)
        variable_summares(conv4, 'conv4/out', 1)
        with tf.variable_scope('vsion'):
            layer1_image1 = conv4[0:1, :, :, 0:16]
            layer1_image1 = tf.transpose(layer1_image1, perm=[3, 1, 2, 0])
            tf.summary.image("filtered_images_layer1", layer1_image1, max_outputs=5)

    with tf.variable_scope('Max_Pooling4') as scope:
        pool4 = tf.nn.max_pool(conv4, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID',
                               name='pooling4')
        norm4 = tf.nn.lrn(pool4, depth_radius=4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name='norm4')

        with tf.variable_scope('vsion'):
            layer1_image1 = norm4[0:1, :, :, 0:16]
            layer1_image1 = tf.transpose(layer1_image1, perm=[3, 1, 2, 0])
            tf.summary.image("filtered_images_layer1", layer1_image1, max_outputs=5)

    with tf.variable_scope('conv5') as scope:
        conv5_w = get_weight([3, 3, 512, 512], regularizer)
        # conv5_w = tf.Variable(tf.truncated_normal(shape=[3, 3, 512, 512], stddev=1.0, dtype=tf.float32),
        #                      name='weights', dtype=tf.float32)
        conv5_b = tf.Variable(tf.constant(value=0.1, dtype=tf.float32, shape=[512]),
                              name='biases', dtype=tf.float32)
        conv = tf.nn.conv2d(norm4, conv5_w, strides=[1, 1, 1, 1], padding='SAME')
        pre_activation = tf.nn.bias_add(conv, conv5_b)
        conv5 = tf.nn.relu(pre_activation, name=scope.name)

        variable_summares(conv5_w, 'conv5/W', None)
        variable_summares(conv5_b, 'conv5/b', None)
        variable_summares(conv5, 'conv5/out', 1)
        with tf.variable_scope('vsion'):
            layer1_image1 = conv5[0:1, :, :, 0:16]
            layer1_image1 = tf.transpose(layer1_image1, perm=[3, 1, 2, 0])
            tf.summary.image("filtered_images_layer1", layer1_image1, max_outputs=5)

    with tf.variable_scope('Max_Pooling5') as scope:
        pool5 = tf.nn.max_pool(conv5, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID',
                               name='pooling5')
        norm5 = tf.nn.lrn(pool5, depth_radius=4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name='norm5')

        with tf.variable_scope('vsion'):
            layer1_image1 = norm5[0:1, :, :, 0:16]
            layer1_image1 = tf.transpose(layer1_image1, perm=[3, 1, 2, 0])
            tf.summary.image("filtered_images_layer1", layer1_image1, max_outputs=5)

    # 全连接层结构，先参考博主
    with tf.variable_scope('local6') as scope:
        reshape = tf.reshape(norm5, shape=[batch_size, -1])
        # 得到每行(每行就是一个图片样本)有多少参数，即共有batch_size行
        dim = reshape.get_shape()[1].value
        local6_w = tf.Variable(tf.truncated_normal(shape=[dim, 128], stddev=0.005, dtype=tf.float32),
                               name='weights', dtype=tf.float32)
        local6_b = tf.Variable(tf.constant(value=0.1, dtype=tf.float32, shape=[128]),
                               name='biases', dtype=tf.float32)
        local6 = tf.nn.relu(tf.matmul(reshape, local6_w) + local6_b, name=scope.name)

        variable_summares(local6_w, 'local6/W', None)
        variable_summares(local6_b, 'local6/b', None)
        variable_summares(local6, 'local6', None)

    with tf.variable_scope('local7') as scope:
        local7_w = tf.Variable(tf.truncated_normal(shape=[128, 128]))
        local7_b = tf.Variable(tf.constant(value=0.1, dtype=tf.float32, shape=[128]),
                               name='biases', dtype=tf.float32)
        local7 = tf.nn.relu(tf.matmul(local6, local7_w) + local7_b, name=scope.name)

        variable_summares(local7_w, 'local7/W', None)
        variable_summares(local7_b, 'local7/b', None)
        variable_summares(local7, 'local7', None)

    # 输出层
    with tf.variable_scope('out_linear') as scope:
        out_w = tf.Variable(tf.truncated_normal(shape=[128, n_clasess], stddev=0.005, dtype=tf.float32),
                            name='out_linear', dtype=tf.float32)
        out_b = tf.Variable(tf.constant(value=0.1, dtype=tf.float32, shape=[n_clasess]),
                            name='biases', dtype=tf.float32)
        out_linear = tf.add(tf.matmul(local7, out_w), out_b, name=scope.name)

        variable_summares(out_w, 'out/W', None)
        variable_summares(out_b, 'out/b', None)
        variable_summares(out_linear, 'out', None)
        print('out_liner = ', out_linear)
    return out_linear


# loss计算，看收藏的CSDN
def losses(logits, labels, regularizer):
    with tf.variable_scope('loss') as scope:
        # 这个才是softmax输出层，以及计算Crossw-Entropy,即交叉熵损失函数的损失值。
        # 将labels转化为one-hot,再将logits与labels进行交叉熵损失函数
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels,
                                                                       name='xentropy_per_example')
        # 默认计算所有元素的均值，对batch_size里每个样本的loss求平均，计算最后的cost值
        loss = tf.reduce_mean(cross_entropy, name='loss')
        if regularizer != None:
            loss = loss + tf.add_n(tf.get_collection('losses'))
            print('loss.type= ', loss)
        tf.summary.scalar(scope.name + '/loss', loss)
    return loss


# loss损失值优化学习，更新权重.
def training(loss, learing_rate):
    with tf.name_scope('optimizer'):
        optimizer = tf.train.AdamOptimizer(learning_rate=learing_rate)
        # 默认trainable=True,则新建变量添加到optimizer优化器中进行优化，若为False,则不进行优化更新。
        global_step = tf.Variable(0, name='global_step', trainable=False)
        # 每运行一次，global_step自动加1
        train_op = optimizer.minimize(loss, global_step=global_step)
    return train_op


# 评估/准确率计算
def evaluation(logits, labels):
    with tf.variable_scope('accuracy') as scope:
        correct = tf.nn.in_top_k(logits, labels, 1)  # 也可以选择tf.equal函数。
        correct = tf.cast(correct, tf.float32)
        accuracy = tf.reduce_mean(correct)
        tf.summary.scalar('/accuracy', accuracy)
        print('T3_accuracy = ', accuracy)
    return accuracy
