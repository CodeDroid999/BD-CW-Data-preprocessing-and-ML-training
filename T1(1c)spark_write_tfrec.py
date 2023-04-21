import tensorflow as tf
import pyspark.sql.functions as F
import pyspark.sql.types as T

# Define the function to convert an image file to TFRecord format
def image_to_tfrecord(row):
    image_path = row.image_path
    label = row.label
    
    # Read the image file
    image = tf.io.read_file(image_path)
    
    # Decode the image file
    image = tf.io.decode_image(image)
    
    # Convert the image to a byte string
    image_bytes = tf.io.serialize_tensor(image)
    
    # Create a feature dictionary
    feature = {
        "image": tf.train.Feature(bytes_list=tf.train.BytesList(value=[image_bytes.numpy()])),
        "label": tf.train.Feature(int64_list=tf.train.Int64List(value=[label])),
    }
    
    # Create an example object
    example = tf.train.Example(features=tf.train.Features(feature=feature))
    
    # Serialize the example object to a string
    serialized_example = example.SerializeToString()
    
    return serialized_example

# Define the schema of the DataFrame
schema = T.StructType([
    T.StructField("image_path", T.StringType()),
    T.StructField("label", T.IntegerType()),
])

# Load the image DataFrame
image_df = spark.read.schema(schema).csv("image_data.csv")

# Convert the image DataFrame to TFRecord format
tfrecord_rdd = image_df.rdd.map(image_to_tfrecord)

# Save the TFRecord files to disk
tfrecord_rdd.saveAsTextFile("tfrecord_data")
