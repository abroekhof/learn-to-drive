import csv
import random

from keras.preprocessing.image import Iterator
from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import backend as K
from keras.utils.np_utils import convert_kernel

import numpy as np
import matplotlib.image as mpimg
import cv2
import h5py
from PIL import Image

img_height = 224
img_width = 224

def save_bottlebeck_features():
    # build the VGG16 network
    model = Sequential()
    model.add(ZeroPadding2D((1, 1), input_shape=(img_height, img_width, 3)))
    model.add(Convolution2D(64, 3, 3, activation='relu', name='conv1_1'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(64, 3, 3, activation='relu', name='conv1_2'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(128, 3, 3, activation='relu', name='conv2_1'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(128, 3, 3, activation='relu', name='conv2_2'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(256, 3, 3, activation='relu', name='conv3_1'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(256, 3, 3, activation='relu', name='conv3_2'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(256, 3, 3, activation='relu', name='conv3_3'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu', name='conv4_1'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu', name='conv4_2'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu', name='conv4_3'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu', name='conv5_1'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu', name='conv5_2'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu', name='conv5_3'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    f = h5py.File('data/vgg16_weights.h5')
    for k in range(f.attrs['nb_layers']):
        if k >= len(model.layers) - 1:
            # we don't look at the last two layers in the savefile (fully-connected and activation)
            break
        g = f['layer_{}'.format(k)]
        weights = [g['param_{}'.format(p)] for p in range(g.attrs['nb_params'])]
        layer = model.layers[k]

        if layer.__class__.__name__ in [
                'Convolution1D', 'Convolution2D', 'Convolution3D', 'AtrousConvolution2D']:
            weights[0] = np.transpose(weights[0], (2, 3, 1, 0))

        layer.set_weights(weights)

    f.close()

    train_set, validation_set = create_data()

    print("Creating training bottleneck features.")
    bottleneck_features_train = model.predict_generator(data_generator(train_set), len(train_set))
    np.save(open('bottleneck_features_train.npy', 'wb'), bottleneck_features_train)

    print("Creating validation bottleneck features.")
    bottleneck_features_validation = model.predict_generator(
        data_generator(validation_set), len(validation_set))
    np.save(open('bottleneck_features_validation.npy', 'wb'), bottleneck_features_validation)

def create_data():
    with open('data/driving_log.csv') as csvfile:
        reader = csv.DictReader(csvfile)
        rows = list(reader)

    random.seed(0)
    random.shuffle(rows)
    split = int(len(rows)/10)
    train_data = rows[split:]
    validation_data = rows[:split]
    return train_data, validation_data

def train_top_model():
    train_set, validation_set = create_data()

    train_data = np.load(open('bottleneck_features_train.npy', 'rb'))
    train_labels = np.array([row['steering'] for row in train_set])

    validation_data = np.load(open('bottleneck_features_validation.npy', 'rb'))
    validation_labels = np.array([row['steering'] for row in validation_set])

    model = Sequential()
    model.add(Flatten(input_shape=train_data.shape[1:]))
    model.add(Dense(256, activation='tanh'))
    model.add(Dropout(0.5))
    model.add(Dense(64, activation='tanh'))
    model.add(Dropout(0.5))
    model.add(Dense(1))

    model.compile(loss='mse', optimizer='adam')

    model.fit(train_data, train_labels,
              nb_epoch=50, batch_size=32,
              validation_data=(validation_data, validation_labels))
    model.save_weights('bottleneck_fc_model.h5')

def preprocess(image):
    # Convert to grayscale.
    # image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
#     image = image.astype(float)
#     # image = np.expand_dims(image, axis=2)
#     # Subtract the mean.
#     image -= np.mean(image)
#     # Normalize.
#     image /= (np.std(image) + 1e-7)
    # Crop hood and horizon out of image.
    image = image.crop((0, 55, 320, 140))
    image = image.resize((224, 224), resample=Image.BILINEAR)
    image = np.asarray(image)
    image = image.astype(np.float32, copy=False)
    return image

def data_generator(rows, batch_size=16):
    iterator = Iterator(len(rows), batch_size=batch_size, shuffle=True, seed=None)
    for index_array, _, batch_size in iterator.index_generator:
        batch_x = np.zeros((batch_size, img_height, img_width, 3))
        batch_y = np.zeros((batch_size, 1))
        for idx, index in enumerate(index_array):
            row = rows[index]
            image = Image.open('data/'+row['center'])
            batch_x[idx] = preprocess(image)
            batch_y[idx] = row['steering']
        yield (batch_x, batch_y)

def create_model():
    model = Sequential()
    model.add(Convolution2D(32, 3, 3, input_shape=(img_height, img_width, 3)))
    model.add(Activation('tanh'))
    model.add(MaxPooling2D((2, 2)))

    model.add(Convolution2D(32, 3, 3))
    model.add(Activation('tanh'))
    model.add(MaxPooling2D((2, 2)))

    model.add(Convolution2D(64, 3, 3))
    model.add(Activation('tanh'))
    model.add(MaxPooling2D((2, 2)))

    model.add(Flatten())
    model.add(Dense(64))
    model.add(Activation('tanh'))
    model.add(Dropout(0.5))
    model.add(Dense(1))

    model.compile(loss='mse', optimizer='adam')
    return model

def train_model(nb_epoch):
    train_set, validation_set = create_data()

    model = create_model()
    model.fit_generator(
        data_generator(train_set),
        samples_per_epoch=len(train_set),
        nb_epoch=nb_epoch,
        validation_data=data_generator(validation_set),
        nb_val_samples=len(validation_set))

    model.save_weights('first_try.h5')


if __name__ == "__main__":
    save_bottlebeck_features()
    train_top_model()
    # train_model(10)