import tensorflow as tf

num_shards = 20
instance_per_shard = 2
for i in range(num_shards):
    filename = ('Coord/data.tfrecords-%.5d-of-%.5d' % (i, num_shards))
    witer = tf.python_io.TFRecordWriter(filename)
    for j in range(instance_per_shard):
        example = tf.train.Example(features=tf.train.Features(feature={
            'i': tf.train.Feature(int64_list=tf.train.Int64List(value=[i])),
            'j': tf.train.Feature(int64_list=tf.train.Int64List(value=[j]))}))
        witer.write(example.SerializeToString())
    witer.close()
