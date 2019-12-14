import tensorflow as tf
from os import listdir
import pandas as pd
import numpy as np

filenames = np.array([x for x in listdir('data/train')])
df = pd.read_csv('data/train_labels.csv')

# here we sort the labels and delete the filenames
df = df.sort_values(by=['id'])
df = df.drop(['id'], axis=1)
labels = df.values
print(labels)
print(type(labels))
print(filenames)