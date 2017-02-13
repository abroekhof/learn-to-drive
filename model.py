
import csv
import random
import os
from keras.models import Sequential
from keras.layers import Convolution2D, ZeroPadding2D, MaxPooling2D
from keras.layers.core import Flatten, Dense, Dropout, Lambda
from keras import backend as K
from keras.optimizers import SGD
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.preprocessing.image import Iterator
from keras import initializations
from keras.callbacks import ModelCheckpoint
import numpy as np
import scipy
from matplotlib import colors
import tensorflow as tf

IMG_HEIGHT = 200
IMG_WIDTH = 100

USE_FLIPPED = True

def create_data():
    print("Creating data")
    rows = []
    with open('data/driving_log.csv') as csvfile:
        reader = csv.reader(csvfile)
        rows = list(reader)

    random.seed(999)
    # randomize rows so that validation and training don't have continuous
    # parts of the track
    random.shuffle(rows)
    data = []
    # separate out zero steering angle examples, so they can be trimmed
    data_zero = []
    for row in rows:
        img_pos = row[0].find('IMG')
        filepath = os.path.join('data', row[0][img_pos:])
        steering = float(row[-4])
        datapoint = {'img': filepath, 'steering': steering, 'flip': False}
        datapoints = [datapoint]
        if USE_FLIPPED:
            # augment the data with a flipped image, reversing the steering as well
            dp_flip = {'img': filepath, 'steering': -steering, 'flip': True}
            datapoints.append(dp_flip)
        if steering == 0:
            data_zero.extend(datapoints)
        else:
            data.extend(datapoints)

    num_zero_angles = int(len(data_zero)*.25)
    rows = data_zero[:num_zero_angles]+data
    random.shuffle(rows)

    # 80/20 training/validation split
    split = int(len(rows)*.1)
    train_data = rows[split:]
    print(len(train_data))
    validation_data = rows[:split]
    print(len(validation_data))
    print("Done creating data")
    return train_data, validation_data

def preprocess(img):
    # Crop horizon out of image.
    #img = img[60:, :, :]
    img = scipy.misc.imresize(img, (IMG_HEIGHT, IMG_WIDTH))
    img = img.astype('float32')
    # normalize the image
    img = img/255.0
    return img

def data_generator(rows, batch_size=128):
    iterator = Iterator(len(rows), batch_size=batch_size, shuffle=True, seed=None)
    for index_array, _, batch_size in iterator.index_generator:
        batch_x = np.zeros((batch_size, IMG_HEIGHT, IMG_WIDTH, 3))
        batch_y = np.zeros((batch_size, 1))
        for idx, index in enumerate(index_array):
            row = rows[index]
            img = scipy.ndimage.imread(row['img'])
            if row['flip']:
                img = np.fliplr(img)
            batch_x[idx] = preprocess(img)
            batch_y[idx] = row['steering']
        yield (batch_x, batch_y)


def steering_model():
    model = Sequential()
    model.add(Convolution2D(24, 5, 5,
                            subsample=(2, 2),
                            name='conv1_1',
                            input_shape=(IMG_HEIGHT, IMG_WIDTH, 3),
                            activation='relu'))
    model.add(Convolution2D(36, 5, 5, subsample=(2, 2), name='conv2_1',
                            activation='relu'))
    model.add(Convolution2D(48, 5, 5, subsample=(2, 2), name='conv3_1',
                            activation='relu'))
    model.add(Convolution2D(64, 3, 3, subsample=(1, 1), name='conv4_1',
                            activation='relu'))
    model.add(Convolution2D(64, 3, 3, subsample=(1, 1), name='conv4_2',
                            activation='relu'))
    model.add(Flatten())
    model.add(Dense(1164, name="dense_0",
                    activation='relu'))
    model.add(Dropout(.5))
    model.add(Dense(100, name="dense_1",
                    activation='relu'))
    model.add(Dense(50, name="dense_2",
                    activation='relu'))
    model.add(Dense(10, name="dense_3",
                    activation='relu'))
    model.add(Dense(1, name="dense_4"))
    return model

def train_model():
    model = steering_model()
    model.compile(loss='mse', optimizer='Adam')

    json_string = model.to_json()
    with open('model.json', mode='w') as outfile:
        outfile.write(json_string)

    train_set, validation_set = create_data()
    model.fit_generator(
        data_generator(train_set),
        samples_per_epoch=len(train_set),
        nb_epoch=40,
        validation_data=data_generator(validation_set),
        nb_val_samples=len(validation_set),
        callbacks=[
            ModelCheckpoint('model.h5', verbose=2, save_weights_only=True, save_best_only=True)
            ]
        )

if __name__ == "__main__":
    train_model()
