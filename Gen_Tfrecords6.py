from PIL import Image
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import Gen_Tfrecords3
import Gen_Tfrecordl2_6

BATCH_SIZE = 32
CAPACITY = 128  # CAPACITY=1000+3*BATCH_SIZE
IMG_W = 400
IMG_H = 400
N_CLASSES = 5
val_batch_total = int(250 / BATCH_SIZE) + 1
loss_total = 0
accrucy_total = 0
val_dir = 'gen_inputData/val/photo'
ckpt_train_dir = 'gen_inputData/train/ckpt/ckpt_train'

val_all_acc = []
val_all_loss = []


def avge(list):
    s = 0
    Len = len(list)
    for i in list:
        s += sum(i)
    s = s / Len
    return s


# 测试图片
val, val_label = Gen_Tfrecordl2_6.get_files(val_dir, 3)
# 测试集数据及标签
print('val.shape= ', np.shape(val))
val_batch, val_label_batch = \
    Gen_Tfrecordl2_6.get_batch(val, val_label, IMG_W, IMG_H, BATCH_SIZE, CAPACITY)
print('val_batch.shape= ', np.shape(val_batch))

'''
regularizer = tf.contrib.layers.l2_regularizer(0.000001)
logits = Gen_Tfrecords3.inference(val_batch, BATCH_SIZE, N_CLASSES, regularizer)
val_loss = Gen_Tfrecords3.losses(logits, val_label_batch, regularizer)
val_acc = Gen_Tfrecords3.evaluation(logits, val_label_batch)
'''

with tf.Session() as sess:
    print('Reading checkpoints....')
    '''ckpt = tf.train.get_checkpoint_state(ckpt_train_dir)
    # 下面两个表示是一样的效果。
    print('ckpt.path= ', ckpt.model_checkpoint_path)
    print('ckpt.path= ', tf.train.latest_checkpoint(ckpt_train_dir))
    # saver = tf.train.Saver()
    # tf.train.Saver()与tf.train.import_meta_graph区别
    # https://blog.csdn.net/sinat_36618660/article/details/98665482'''
    # tf.train.Saver() 1,保存训练模型，以及所有参数；2，只加载w,b等训练的数据，其他不加载；
    # tf.train.import_meta_graph加载meta中的图，以及图上定义的所有参数，w,b以及中间参数；
    # 所以保存用tf.train.Saver()，加载用tf.train.import_meta_graph()。
    saver = tf.train.import_meta_graph(ckpt_train_dir + '/my_model.ckpt-5.meta')
    saver.restore(sess, tf.train.latest_checkpoint(ckpt_train_dir))

    # 获取参数
    print('********* 获取参数 ***********')
    graph = tf.get_default_graph()  # 获取当前默认计算图
    conv1_weights = graph.get_tensor_by_name("conv1/weights:0")
    # print(sess.run(conv1_weights))
    # print(sess.run('tr_val_label:0'))
    # 反正下边这两行要加，具体啥作用咱也不太懂。
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    try:
        print('****** 开始测试 ******')
        for iteration in range(val_batch_total):
            val_batch1, val_label_batch1 = sess.run([val_batch, val_label_batch])  # 获取X,Y。
            feed_dict = {"tr_batch:0": val_batch1, "tr_label_batch:0": val_label_batch1}  # 将输入传入给占位符。

            # 按照训练模型Gen_Tfrecords4.py获取 predict 方法并用新的数据feed_dict，获取val_logits。
            val_logits = tf.get_collection("predict")
            val_acc = tf.get_collection("acc")
            val_loss = tf.get_collection("loss")

            val_pred, val_acc, val_loss = sess.run([val_logits, val_acc, val_loss], feed_dict)
            val_all_acc.append(val_acc)
            val_all_loss.append(val_loss)

        print('val_pred.shape= ', np.shape(val_pred))
        avge_acc = avge(val_all_acc)
        avge_loss = avge(val_all_loss)
        print('avge_acc= ', avge_acc)
        print('avge_loss= ', avge_loss)

    except tf.errors.OutOfRangeError:  # 如果读取到文件队列末尾会抛出此异常
        print("done! now lets kill all the threads……")
    finally:
        # 协调器coord发出所有线程终止信号
        coord.request_stop()
        print('all threads are asked to stop!')
