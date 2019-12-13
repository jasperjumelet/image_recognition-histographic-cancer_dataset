#from image_preprocessing import imageToPixel
import pandas as pd
from matplotlib import image
from os import listdir
import pickle
import numpy as np
from sklearn.externals import joblib

# load in all image_values from a directory of images
def imageToPixel():
    loaded_images = np.empty((0, 96,96,3))

    for filename in listdir('data/train'):
            # load image
            img_data = image.imread('data/train/' + filename)
            # store loaded image
            img_data = img_data.reshape(1, 96, 96, 3)
            loaded_images = np.concatenate((loaded_images, img_data), axis=0)

            # loaded_images.append(img_data)
            #loaded_images.extend(img_data)
    print(loaded_images.shape)
    return loaded_images

# here we make sure that all the data is splitted into X_train, y_train, X_test, y_test
def dataPreprocessing(split):
    df = pd.read_csv('data/train_labels.csv')

    # we use sort here because our os sort the picture names and we need to match them with the right label
    df = df.sort_values(by=['id'])
    
    # we drop id (picture name) because it will hinder our training model
    df = df.drop(['id'], axis=1)

    # here we call all the image values
    image_values = imageToPixel()
    
    
    # here we split the data
    X_train = image_values[:(len(image_values) * split // 100)]
    y_train = image_values[(len(image_values) * split // 100):]

    X_test = df.iloc[:int(df.shape[0] * split // 100), -1].values
    y_test = df.iloc[int(df.shape[0] * split // 100):, -1].values

    print(type(X_train))
    return X_train, y_train, X_test, y_test

# X_train, y_train, X_test, y_test = dataPreprocessing(80)
# print("Shape of x_train = ", X_train[0].shape)

# here we saved our created data to pickle files
def saveToPickle(split_percentage):
    X_train, y_train, X_test, y_test = dataPreprocessing(split_percentage)

    # with open('train.pickle', 'wb') as f:
    #     pickle.dump([X_train, y_train], f)

    # with open('test.pickle', 'wb') as f:
    #     pickle.dump([X_test, y_test], f)
    joblib.dump(X_train, 'X_train.pkl')
    joblib.dump(y_train, 'y_train.pkl')
    joblib.dump(X_test, 'X_test.pkl')
    joblib.dump(y_test, 'y_test.pkl')

saveToPickle(80)