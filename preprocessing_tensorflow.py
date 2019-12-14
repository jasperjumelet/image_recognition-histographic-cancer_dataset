import tensorflow as tf
from os import listdir
import numpy as np

filenames = np.array([x for x in listdir('data/train')])

print(filenames)