import csv
import random

from keras.preprocessing.image import Iterator
from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import backend as K
from keras.utils.np_utils import convert_kernel
from keras.applications.vgg16 import VGG16

import numpy as np
import matplotlib.image as mpimg
import cv2
import h5py
from PIL import Image

img_height = 224
img_width = 224

TOP_MODEL_WEIGHTS_PATH = 'top_model.h5'
VGG_MODEL_WEIGHTS_PATH = 'data/vgg16_weights.h5'

def create_vgg_model():
    print("Creating pre-trained model")
    model = VGG16(weights='imagenet', include_top=False)
    print("Done creating pre-trained model")
    return model

def create_top_model(input_shape):
    model = Sequential()
    model.add(Flatten(input_shape=input_shape))
    model.add(Dense(256, activation='tanh'))
    model.add(Dropout(0.5))
    model.add(Dense(64, activation='tanh'))
    model.add(Dropout(0.5))
    model.add(Dense(1))
    return model

def save_bottlebeck_features():
    model = create_vgg_model()

    train_set, validation_set = create_data()

    print("Creating training bottleneck features.")
    bottleneck_features_train = model.predict_generator(data_generator(train_set), len(train_set))
    np.save(open('bottleneck_features_train.npy', 'wb'), bottleneck_features_train)

    print("Creating validation bottleneck features.")
    bottleneck_features_validation = model.predict_generator(
        data_generator(validation_set), len(validation_set))
    np.save(open('bottleneck_features_validation.npy', 'wb'), bottleneck_features_validation)

def create_data():
    print("Creating data")
    with open('data/driving_log.csv') as csvfile:
        reader = csv.DictReader(csvfile)
        rows = list(reader)

    random.seed(0)
    random.shuffle(rows)
    split = int(len(rows)/10)
    train_data = rows[split:]
    validation_data = rows[:split]
    print("Done creating data")
    return train_data, validation_data

def train_top_model():
    train_set, validation_set = create_data()

    train_data = np.load(open('bottleneck_features_train.npy', 'rb'))
    train_labels = np.array([row['steering'] for row in train_set])

    validation_data = np.load(open('bottleneck_features_validation.npy', 'rb'))
    validation_labels = np.array([row['steering'] for row in validation_set])

    model = create_top_model(input_shape=train_data[0].shape)
    model.compile(loss='mse', optimizer='adam')
    model.fit(train_data, train_labels,
              nb_epoch=10, batch_size=32,
              validation_data=(validation_data, validation_labels))
    model.save_weights(TOP_MODEL_WEIGHTS_PATH)

def finetune_model():
    model = create_vgg_model()

    top_model = create_top_model()
    top_model.load_weights(TOP_MODEL_WEIGHTS_PATH)

    model.add(top_model)
    for layer in model.layers[:25]:
        layer.trainable = False

    model.compile(loss='mse', optimizer='adam')

    train_set, validation_set = create_data()
    model.fit_generator(
        data_generator(train_set),
        samples_per_epoch=len(train_set),
        nb_epoch=10,
        validation_data=data_generator(validation_set),
        nb_val_samples=len(validation_set))
    model.save_weights('model.h5')
    json_string = model.to_json()
    with open('model.json', mode='w') as f:
        f.write(json_string)


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

def data_generator(rows, batch_size=32):
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



if __name__ == "__main__":
    save_bottlebeck_features()
    train_top_model()
    finetune_model()
