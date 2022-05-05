import os
import tensorflow as tf
from PIL import Image

orig_picture = 'photo1/'
classes = ['AoKeng', 'LieWen', 'WeiHantou', 'WeiRonghe', 'WuQuexian']
gen_picture = 'gen_inputData/train/photo1/'
examples = []
lables = []


def creat_record():
    writer = tf.python_io.TFRecordWriter('gen_inputData/moi_train.tfrecords')
    num_img = 0
    for index, name in enumerate(classes):
        classes_path = orig_picture + name + "/"
        print('class= ', name)
        print('index = ', index)
        for img_name in os.listdir(classes_path):
            img_path = classes_path + img_name
            img = Image.open(img_path)
            img = img.resize((400, 400))
            img_raw = img.tobytes()
            # print(index, img_raw)
            example = tf.train.Example(features=tf.train.Features(feature={
                "label": tf.train.Feature(int64_list=tf.train.Int64List(value=[index])),
                "img_raw": tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw]))
            }))
            writer.write(example.SerializeToString())
            num_img = num_img + 1
    # print("num_img= ", num_img)
    writer.close()
    return num_img


# 读取TFRecords文件数据，输出格式一致（大小）的图像及标签列表
def read_and_decode(filename):
    # filename:保存TFRecords的文件地址；
    # 读取TFRecords文件数据，并生成列表，可设置是否打乱列表及读取次数等。
    filename_queue = tf.train.string_input_producer([filename])
    # 读取队列信息
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    # 解析读取的队列信息
    features = tf.parse_single_example(serialized_example, features={
        'label': tf.FixedLenFeature([], tf.int64),
        'img_raw': tf.FixedLenFeature([], tf.string)})
    label, img = features['label'], features['img_raw']
    # 从原始图像数据解析出像素矩阵，并根据原图像尺寸还原图像,所以下边尺寸和上边保存时保持一样，(224,224,3)
    img = tf.decode_raw(img, tf.uint8)
    img = tf.reshape(img, [400, 400, 3])
    # 图像矩阵归一化，放到后面网络输入时处理也可以,-0.5啥意思？
    # img=tf.cast(img,float32)/255-0.5
    # label = tf.cast(label, tf.int32)
    return img, label


def get_out(file_dire, batch, label):
    for img_name in os.listdir(file_dire):
        print("img_name is ", img_name)


if __name__ == '__main__':
    num_img = creat_record()
    batch = read_and_decode('gen_inputData/moi_train.tfrecords')
    init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
    with tf.Session() as sess:
        sess.run(init_op)
        # 一般有队列queue,就会有多线程下面的两行
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
        for i in range(num_img):
            # 每次取一个
            example, lab = sess.run(batch)
            # 每个样本建一个文件并保存
            if lab == 0:
                img = Image.fromarray(example, 'RGB')
                img.save(gen_picture + '/AoKeng/' + str(i) + 'samples' + str(lab) + '.jpg')
            elif lab == 1:
                img = Image.fromarray(example, 'RGB')
                img.save(gen_picture + '/LieWen/' + str(i) + 'samples' + str(lab) + '.jpg')
            elif lab == 2:
                img = Image.fromarray(example, 'RGB')
                img.save(gen_picture + '/WeiHantou/' + str(i) + 'samples' + str(lab) + '.jpg')
            elif lab == 3:
                img = Image.fromarray(example, 'RGB')
                img.save(gen_picture + '/WeiRonghe/' + str(i) + 'samples' + str(lab) + '.jpg')
            elif lab == 4:
                img = Image.fromarray(example, 'RGB')
                img.save(gen_picture + '/WuQuexian/' + str(i) + 'samples' + str(lab) + '.jpg')

            # 保存在一个文件夹
            '''img = Image.fromarray(example, 'RGB')
            img.save(gen_picture + '/all_flowers/' + str(i) +'#'+ str(lab)+'#' + '.jpg')'''

        coord.request_stop()
        coord.join(threads)
        sess.close()
