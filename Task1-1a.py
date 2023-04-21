

# Task1: 1a) Create the script 
# steps in code
# importing libraries and modules
# define the mapping functions get_label, resize, and recompress as before.
# define the function write_tfrecords to write TFRecord files to the cloud using Spark. This function takes an index and an iterator as input, where the iterator contains tuples of filename and image contents. It returns a list of the names of the created TFRecord files.
# initialize Spark by creating a SparkConf and SparkContext, and a SparkSession from the SparkContext.
# load the image filenames into an RDD using sc.binaryFiles.
# sample the RDD to a smaller number using sample.
# preprocess the images and write the TFRecord files using map, mapPartitionsWithIndex, and write_tfrecords.
# collect the list of created TFRecord files using collect.
# print the list of created TFRecord files.



# importing modules and libraries
import os
import tensorflow as tf
import numpy as np
from PIL import Image
from pyspark import SparkConf, SparkContext
from pyspark.sql import SparkSession

# Define mapping functions
def get_label(path):
    """
    Returns the label of an image path.
    """
    return os.path.basename(os.path.dirname(path))

def resize(path, size):
    """
    Resizes an image given its path and a target size.
    Returns the resized image as a NumPy array.
    """
    with Image.open(path) as img:
        img = img.resize(size)
        return np.asarray(img)

def recompress(img, format):
    """
    Recompresses an image given its contents as a NumPy array and a target format.
    Returns the contents of the recompressed image as a byte string.
    """
    with Image.fromarray(img) as img:
        img = img.convert(format)
        with tf.io.BytesIO() as output:
            img.save(output, format=format)
            contents = output.getvalue()
            return contents

# Define function to write TFRecord files
def write_tfrecords(index, iterator):
    """
    Writes TFRecord files given an index and an iterator of (filename, contents) tuples.
    Returns a list of the names of the created TFRecord files.
    """
    tfrecord_files = []
    for i, item in enumerate(iterator):
        filename, contents = item
        if i % 100 == 0:
            print(f'Writing record {i}')
        tfrecord_file = f'flowers_{index}_{i:05d}.tfrecord'
        tfrecord_files.append(tfrecord_file)
        with tf.io.TFRecordWriter(tfrecord_file) as writer:
            example = tf.train.Example(features=tf.train.Features(feature={
                'image': tf.train.Feature(bytes_list=tf.train.BytesList(value=[contents])),
                'label': tf.train.Feature(bytes_list=tf.train.BytesList(value=[get_label(filename).encode('utf-8')]))
            }))
            writer.write(example.SerializeToString())
    return tfrecord_files

# Initialize Spark
conf = SparkConf().setAppName('preprocessing')
sc = SparkContext(conf=conf)
spark = SparkSession(sc)

# Load image filenames into RDD
filenames_rdd = sc.binaryFiles('gs://path/to/flowers/*/*.jpg')

# Sample the RDD
sampled_rdd = filenames_rdd.sample(False, 0.02)

# Preprocess images and write TFRecord files
tfrecord_files_rdd = sampled_rdd.map(lambda x: (x[0], resize(x[1].open(), (180, 180)))).map(lambda x: (x[0], recompress(x[1], 'JPEG'))).mapPartitionsWithIndex(write_tfrecords)

# Collect list of created TFRecord files
tfrecord_files = tfrecord_files_rdd.collect()

# Print list of created TFRecord files
print(tfrecord_files)
