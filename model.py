#!/bin/env python

"""Self-driving car nanodegree on Udacity; Project 3: Behaviour cloning
"""

import csv
import os

from keras.layers.convolutional import Convolution2D, Cropping2D, MaxPooling2D, ZeroPadding2D
from keras.layers.core import Activation, Dense, Dropout, Flatten, Lambda
from keras.models import Sequential
# import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from sklearn import model_selection
from sklearn import utils
import tensorflow as tf

# Fix error with TF and Keras
tf.python.control_flow_ops = tf


# COURSES = ['c1', 'c2']
COURSES = ['c0', 'c1s', 'c1n', 'c1r2', 'c1i', 'c1y', 'c2', 'c1z', 'c1o']

CSV_FILENAME = 'driving_log.csv'
IMG_DIRNAME = 'IMG'
CORRECTION = 0.2
BATCH_SIZE = 64
EPOCH = 10


def load_steering_samples(dir_path):
    """Load the steering samples from the CSV file.

    :param dir_path: The path to the directory that contains the CSV file to
                     load.
    
    :returns: The list of the lines of the CSV file contents.
    """
    steering_samples = []

    with open(os.path.join(dir_path, CSV_FILENAME), 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            steering_samples.append(row)

    return steering_samples


def load_data(samples, dir_path='.', batch_size=BATCH_SIZE):
    """Load the data from CSV and images files into the memory.

    This function is the generator to yield the loaded image data in the batch
    size.

    :param samples:    The loaded steering data sample as numpy array.
    :param dir_path:   The path to the directory that contains the CSV file and
                       the directory has image files to load. Defaults to '.'.
    :param batch_size: The size of the batch data to generate defaults to 32.

    :returns: A tuple of the car images and the steering data that contains the
              series of data come from the three angles, center, left and right
              in the order.
    """
    n_samples = len(samples)

    while True:
        samples = utils.shuffle(samples)
        for offset in range(0, n_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]
        
            images = []
            angles = []
            for batch_sample in batch_samples:
                steering_center = float(batch_sample[3])
                steering_left = steering_center + CORRECTION
                steering_right = steering_center - CORRECTION
        
                image_center_path = os.path.join(
                    *batch_sample[0].split('/')[-3:])
                image_center = np.asarray(Image.open(os.path.join(
                    dir_path, image_center_path)))
        
                images.append(image_center)
                angles.append(steering_center)

                image_center_flipped = np.fliplr(image_center)
                images.append(image_center_flipped)
                steering_center_flipped = - steering_center
                angles.append(steering_center_flipped)

                image_left_path = os.path.join(
                    *batch_sample[1].split('/')[-3:])
                if image_left_path:
                    image_left = np.asarray(Image.open(os.path.join(
                        dir_path, image_left_path)))
                    images.append(image_left)
                    angles.append(steering_left)
                    
                    image_left_flipped = np.fliplr(image_left)
                    images.append(image_left_flipped)
                    steering_left_flipped = - steering_left
                    angles.append(steering_left_flipped)

                image_right_path = os.path.join(
                    *batch_sample[2].split('/')[-3:])
                if image_right_path:
                    image_right = np.asarray(Image.open(os.path.join(
                        dir_path, image_right_path)))
                    images.append(image_right)
                    angles.append(steering_right)

                    image_right_flipped = np.fliplr(image_right)
                    images.append(image_right_flipped)
                    steering_right_flipped = - steering_right
                    angles.append(steering_right_flipped)

            X_train = np.asarray(images)
            y_train = np.asarray(angles)

            yield utils.shuffle(X_train, y_train)


if __name__ == '__main__':
    samples = [load_steering_samples(course) for course in COURSES]
    # Flatten the sample data
    samples = [item for sublist in samples for item in sublist]

    train_samples, validation_samples = model_selection.train_test_split(
        samples, test_size=0.3)

    train_data = load_data(train_samples)
    validation_data = load_data(validation_samples)

    # Nvidia's network:
    # http://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf
    model = Sequential()
    # Crop the image data
    model.add(Cropping2D(cropping=((50, 20), (0, 0)), input_shape=(160, 320, 3)))
    # Normalize the image data
    ch, row, col = 3, 90, 320
    model.add(Lambda(lambda x: x/255.0 - 0.5,
                     input_shape=(row, col, ch),
                     output_shape=(row, col, ch)))
    # Convolutional layers
    model.add(Convolution2D(24, 5, 5, subsample=(2, 2)))
    model.add(Activation('relu'))
    model.add(Dropout(0.1))
    model.add(Convolution2D(36, 5, 5, subsample=(2, 2)))
    model.add(Activation('relu'))
    model.add(Dropout(0.1))
    model.add(Convolution2D(48, 5, 5, subsample=(2, 2)))
    model.add(Activation('relu'))
    model.add(Dropout(0.1))
    model.add(Convolution2D(64, 3, 3))
    model.add(Activation('relu'))
    model.add(Dropout(0.1))
    model.add(Convolution2D(64, 3, 3))
    model.add(Activation('relu'))
    model.add(Dropout(0.1))
    # Flatten and dense layers
    model.add(Flatten())
    model.add(Dense(100))
    model.add(Dense(50))
    model.add(Dense(10))
    model.add(Dense(1))

    model.compile('adam', 'mean_squared_error', ['mean_squared_error'])
    history = model.fit_generator(
        train_data, samples_per_epoch=int(len(train_samples)),
        validation_data=validation_data,
        nb_val_samples=int(len(validation_samples)),
        nb_epoch=EPOCH, verbose=1)

    model.save('model.h5')

    print(history.history.keys())

    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model mean squared error loss')
    plt.ylabel('mean squared error loss')
    plt.xlabel('epoch')
    plt.legend(['training set', 'validation set'], loc='upper right')
    plt.show()
