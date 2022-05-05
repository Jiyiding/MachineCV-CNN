import os
import numpy as np
import tensorflow as tf
import Gen_Tfrecords2
import Gen_Tfrecords3
import pandas as pd
import torch

N_CLASSES = 5
IMG_W = 400
IMG_H = 400
BATCH_SIZE = 32
CAPACITY = 64  # 队列中元素的最大数量，大于2倍的BATCH_SIZE就行
MAX_EPOCH = 100
learing_rate = 0.00005
batch_rate = 0
batch_total = int(1850 / BATCH_SIZE)
train_batch_total = int(batch_total * (1 - batch_rate))
val_batch_total = batch_total - train_batch_total
train_dir = 'gen_inputData/train/photo1'  # 训练样本读入路径
ckpt_train_dir = 'gen_inputData/train/ckpt/ckpt_train'  # ckpt_train存储路径
logs_train_dir = 'gen_inputData/train/logs/logs_train'  # logs_train存储路径
list_train_dir = 'gen_inputData/train/list/'  # list_train存储路径


# logs_test_dir = 'gen_inputData/train/logs_test'  # logs_test存储路径




def avge(list):
    sum = 0
    Len = len(list)
    for i in list:
        sum += i
    sum = (sum / Len)
    return sum


def list_to_excel(list, s, epoch):
    dataframe = pd.DataFrame(list)
    dataframe.to_excel(list_train_dir + s + str(epoch) + '_list.xls')


train, train_label, val, val_label = Gen_Tfrecords2.get_files(train_dir, batch_rate)

train_batch, train_label_batch = \
    Gen_Tfrecords2.get_batch(train, train_label, IMG_W, IMG_H, BATCH_SIZE, CAPACITY)
print('train_batch.type', type(train_batch))

# 训练数据的占位符---不知道为啥用下边这个，测试时用下边这个占位符时显示graph中没有此占位符。。
'''
with tf.name_scope('input'):
    tr_batch = tf.placeholder(tf.float32, shape=[BATCH_SIZE, IMG_W, IMG_H, 3], name="tr_batch")
    tr_label_batch = tf.placeholder('int64', shape=[BATCH_SIZE, ], name="tr_label_batch")
    # 通过tf.summary.image函数定义将当前图片信息写入日志的操作。max_outputs表示显示化输出图像数量。
    tf.summary.image('in_input', tr_batch, max_outputs=5)

'''
tr_batch = tf.placeholder(tf.float32, shape=[BATCH_SIZE, IMG_W, IMG_H, 3], name="tr_batch")
tr_label_batch = tf.placeholder('int64', shape=[BATCH_SIZE, ], name="tr_label_batch")

# 正则项
regularizer = tf.contrib.layers.l1_regularizer(0.000001)

train_logits = Gen_Tfrecords3.inference(tr_batch, BATCH_SIZE, N_CLASSES, regularizer)
train_loss = Gen_Tfrecords3.losses(train_logits, tr_label_batch, regularizer)
train_op = Gen_Tfrecords3.training(train_loss, learing_rate)
train_accuracy = Gen_Tfrecords3.evaluation(train_logits, tr_label_batch)

tf.add_to_collection("predict", train_logits)
tf.add_to_collection("acc", train_accuracy)
tf.add_to_collection("loss", train_loss)

# log汇总记录
summary_op = tf.summary.merge_all()
with tf.Session() as sess:
    # 生成一个write来写log文件
    train_write = tf.summary.FileWriter(logs_train_dir, sess.graph)
    # 生成saver来存储训练好的模型
    saver = tf.train.Saver()
    init_op = tf.global_variables_initializer()
    sess.run(init_op)
    # 队列监控
    coord = tf.train.Coordinator()
    # 启动执行文件名队列填充的线程；
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    # 进行batch训练
    try:
        for epoch in np.arange(MAX_EPOCH):
            all_acc = []
            all_loss = []
            if coord.should_stop():
                break
            # 下边这个才是训练一次完整的数据集！
            for iteration in range(train_batch_total):
                train_batch1, train_label_batch1 = sess.run([train_batch, train_label_batch])
                _, tra_loss, tra_accuracy = sess.run([train_op, train_loss, train_accuracy],
                                                     feed_dict={tr_batch: train_batch1,
                                                                tr_label_batch: train_label_batch1})

                all_acc.append(tra_accuracy)
                all_loss.append(tra_loss)
            # 将列表转化成excel形式
            list_to_excel(all_acc, 'acc', epoch)
            list_to_excel(all_loss, 'loss', epoch)

            avge_acc = avge(all_acc)
            avge_loss = avge(all_loss)

            if epoch % 1 == 0:
                # Step是训练全部样本的次数，loss和accuracy是一次batch_size的。
                # 这里是每epoch保存一次
                summary_str = sess.run(summary_op, feed_dict={tr_batch: train_batch1,
                                                              tr_label_batch: train_label_batch1})
                train_write.add_summary(summary_str, epoch)
                # train_loss= %.8f,其中 . 表示小数点后，8表示小数点后保留8位。
                print('Step %d, train_loss= %.8f, train_accuracy= %.2f%%' %
                      (epoch, avge_loss, avge_acc * 100))
                # 配置运行时需要记录的信息pdf
                # run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                # 运行时记录运行信息pdf,错了。。
                # run_metadata = tf.RunMetadata()
                # sess.run(run_options=run_options, run_metadata=run_metadata)
                # train_write.add_run_metadata(run_metadata, 'step%03d' % step)
                # summary_str = sess.run(summary_op)
                # train_write.add_summary(summary_str, epoch)
            # 保存模型
            if (epoch + 1) % 1 == 0:
                checkpoint_path = os.path.join(ckpt_train_dir, 'my_model.ckpt')
                saver.save(sess, checkpoint_path, global_step=epoch)
        train_write.close()

    except tf.errors.OutOfRangeError:  # 如果读取到文件队列末尾会抛出此异常。
        print('Done trianing -- epoch limit reached')
    finally:  # 协调器coord所有线程终止信号
        coord.request_stop()
        print('all threads are asked to stop!')
