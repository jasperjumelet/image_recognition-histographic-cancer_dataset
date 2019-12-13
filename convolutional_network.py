from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, BatchNormalization, Activation
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.constraints import maxnorm
from keras.utils import np_utils
from sklearn.externals import joblib
import numpy as np
import pandas as pd
import os
import pickle

X_train = joblib.load("X_train.pkl")
y_train = joblib.load("y_train.pkl")
X_test = joblib.load("X_test.pkl")
y_test = joblib.load("y_test.pkl")

#X_train = np.asarray(X_train)
print(type(X_train))
# with open('train.pickle', 'rb') as f:
#     X_train , y_train = pickle.load(f)

# with open('test.pickle', 'rb') as f:
#     X_test, y_test = pickle.load(f)

print(X_train[0].shape)
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train = X_train / 255.0
X_test = X_test / 255.0

# print(X_train)
# updated_X_train = np.empty([1,1])
# X_train =X_train.reshape([1,96, 96,3])
# updated_X_train = np.append(updated_X_train, X_train)
# updated_X_train = updated_X_train.reshape([1,96,96,3])

print(X_train)

#print(updated_X_train)
epochs = 25
batch_size = 192
optimizer = 'adam'
class_num = 2

seed = 21

#print(X_train.shape)
# Used this so it works on my device
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
os.environ["CUDA_VISIBLE_DEVICES"]="-1"

model = Sequential()

model.add(Conv2D(96, (3,3), input_shape=(96,96,3), padding='same'))
model.add(Activation('relu'))
model.add(Dropout(0.2))
model.add(BatchNormalization())

model.add(Conv2D(192, (3, 3), padding='same'))
model.add(Activation('relu'))

model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))
model.add(BatchNormalization())

model.add(Conv2D(192, (3, 3), padding='same'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))
model.add(BatchNormalization())
    
model.add(Conv2D(384, (3, 3), padding='same'))
model.add(Activation('relu'))
model.add(Dropout(0.2))
model.add(BatchNormalization())

model.add(Flatten())
model.add(Dropout(0.2))

model.add(Dense(768, kernel_constraint=maxnorm(3)))
model.add(Activation('relu'))
model.add(Dropout(0.2))
model.add(BatchNormalization())

model.add(Dense(384, kernel_constraint=maxnorm(3)))
model.add(Activation('relu'))
model.add(Dropout(0.2))
model.add(BatchNormalization())

model.add(Dense(class_num))
model.add(Activation('softmax'))


# compile the model
model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

print(model.summary())
np.random.seed(seed)

model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=epochs, batch_size=batch_size)

model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)



model.save_weights("trained_weights.h5")