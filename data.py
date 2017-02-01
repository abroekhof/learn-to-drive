import random
import csv
import numpy as np
from PIL import Image
from keras.preprocessing.image import Iterator

IMG_HEIGHT = 299
IMG_WIDTH = 299

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
        batch_x = np.zeros((batch_size, IMG_HEIGHT, IMG_WIDTH, 3))
        batch_y = np.zeros((batch_size, 1))
        for idx, index in enumerate(index_array):
            row = rows[index]
            image = Image.open('data/'+row['center'])
            batch_x[idx] = preprocess(image)
            batch_y[idx] = row['steering']
        yield (batch_x, batch_y)
