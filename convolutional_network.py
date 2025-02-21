from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, BatchNormalization, Activation
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.constraints import maxnorm
from keras.utils import np_utils
from sklearn.externals import joblib
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import pickle
from train_preprocessing_tensorflow import dataset

train_dataset = dataset()

# print("the total size = " + str(len(list(train_dataset))))
DATASET_SIZE = 0

for elem in train_dataset:
      DATASET_SIZE += 1

x_train = train_dataset.take(int(0.8 * DATASET_SIZE))
y_train = train_dataset.skip(int(0.8 * DATASET_SIZE))
#print("The splitted size of x_train = " + str(len(list(x_train))))
#print("The splitted size of y_train = " + str(len(list(x_train))))







def show_batch(image_batch, label_batch):
  plt.figure(figsize=(10,10))
  for n in range(4):
      ax = plt.subplot(2,2,n+1)
      plt.imshow(image_batch[n])
      if label_batch[n] == 1:
            plt.title("Cancer_cell")
      else:
            plt.title("Normal_cell")  
      plt.axis('off')

features, labels = next(iter(train_dataset))
show_batch(features.numpy(), labels.numpy())
plt.show()




#
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

model.fit(x_train, y_train, epochs=epochs)  #validation_data=(X_test, y_test), batch_size=batch_size

model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)



model.save_weights("trained_weights.h5")
