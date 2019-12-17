import tensorflow as tf
from os import listdir
import os
import pandas as pd
import numpy as np

os.environ["CUDA_VISIBLE_DEVICES"]="-1"

filenames = np.array([x for x in listdir('data/train')])

df = pd.read_csv('data/train_labels.csv')

# here we sort the labels and delete the filenames
df = df.sort_values(by=['id'])
df = df.drop(['id'], axis=1)
labels = df.values
labels = np.squeeze(labels)



def parse_function(filename, label):
    image_string = tf.io.read_file("data/train/" + filename)

    # Don't use tf.image.decode_image, or the output shape will be undefined
    # the image decode jpeg need to be fixed 
    image = tf.image.decode_jpeg(image_string, channels=3) 

    #This will convert to float values in [0, 1]
    image = tf.image.convert_image_dtype(image, tf.float32)

    resized_image = tf.image.resize(image, [96, 96])
    return resized_image, label

def train_preprocess(image, label):
    image = tf.image.random_flip_left_right(image)

    image = tf.image.random_brightness(image, max_delta=32.0 / 255.0)
    image = tf.image.random_saturation(image, lower=0.5, upper=1.5)

    # Make sure the image is still in [0, 1]
    image = tf.clip_by_value(image, 0.0, 1.0)

    return image, label

# here we define the batchsize
batch_size = 4

# here we create the dataset piping
dataset = tf.data.Dataset.from_tensor_slices((filenames, labels))
dataset = dataset.shuffle(len(filenames))
dataset = dataset.map(parse_function, num_parallel_calls=4)
dataset = dataset.map(train_preprocess, num_parallel_calls=4)
dataset = dataset.batch(batch_size)
dataset = dataset.prefetch(1)

# here we can iterate over it to see the dataset

# for elem in dataset:
#     print(elem)
