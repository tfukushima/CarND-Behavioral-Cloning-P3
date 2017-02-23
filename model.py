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


# COURSES = ['c1n', 'c2']
COURSES = ['c1n']

CSV_FILENAME = 'driving_log.csv'
IMG_DIRNAME = 'IMG'
CORRECTION = 0.6
BATCH_SIZE = 64
SAMPLES_PER_EPOCH = 32
EPOCH = 5


def process_image(ndarray):
    """Process the numpy array image data.

    Convert the color image data into the grayscale one.

    :returns: The processed image data in numpy array.
    """
    return ndarray


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
                steering_left = steering_center - CORRECTION
                steering_right = steering_center + CORRECTION
        
                image_center_path = os.path.join(
                    *batch_sample[0].split('/')[-3:])
                # image_center = cv2.imread(os.path.join(
                image_center = np.asarray(Image.open(os.path.join(
                    dir_path, image_center_path)))
        
                images.append(image_center)
                angles.append(steering_center)

                # Flip image and the angle
                image_center_flipped = np.fliplr(image_center)
                images.append(image_center_flipped)
                steering_center_flipped = - steering_center
                angles.append(steering_center_flipped)

                image_left_path = os.path.join(
                    *batch_sample[1].split('/')[-3:])
                if image_left_path:
                    # image_left = cv2.imread(os.path.join(
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
                    # image_right = cv2.imread(os.path.join(
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
        
    # car_images = np.array([])
    # steering_angles = np.array([])
    # 
    # for course in COURSE:
    #     with open(os.path.join(course, CSV_FILENAME), 'r') as f:
    #         reader = csv.reader(f)
    #         for row in reader:
    #             steering_center = float(row[3])
    #             steering_left = steering_center + CORRECTION
    #             steering_right = steering_center - CORRECTION
    # 
    #             directory = os.path.join(course, IMG_DIRNAME)
    #             img_center = process_image(np.asarray(Image.open(
    #                 os.path.join(directory, row[0]))))
    #             img_left = process_image(np.asarray(Image.open(
    #                 os.path.join(directory, row[1]))))
    #             img_right = process_image(np.asarray(Image.open(
    #                 os.path.join(directory, row[2]))))
    # 
    #             car_images.append((img_center, img_left, img_right))
    #             steering_angles.append(
    #                 (steering_center, steering_left, steering_right))
    #             
    # return car_images, steering_angles


def mirror_data(loaded_data):
    """Get the mirrored data to mitigate the left curve bias.

    :paarm loaded_data: The tuple of the series of data loaded from the CSV and
                        the image files.
    :returns: 
    """

if __name__ == '__main__':
    samples = [load_steering_samples(course) for course in COURSES]
    # Flatten the sample data
    samples = [item for sublist in samples for item in sublist]

    train_samples, validation_samples = model_selection.train_test_split(
        samples, test_size=0.2)

    train_data = load_data(train_samples)
    validation_data = load_data(validation_samples)

    model = Sequential()
    # Crop the image data
    model.add(Cropping2D(cropping=((50, 20), (0, 0)), input_shape=(160, 320, 3)))
    # Normalize the image data
    ch, row, col = 3, 90, 320
    model.add(Lambda(lambda x: x/127.5 - 1.0,
                     input_shape=(row, col, ch),
                     output_shape=(row, col, ch)))
    # Convolutional layer
    # model.add(Convolution2D(2, 3, 3, border_mode='valid', activation='elu'))
    # model.add(MaxPooling2D((4, 4), (4, 4), 'valid'))
    # model.add(Dropout(0.25))
    # model.add(Flatten())
    # model.add(Dense(1))

    model.add(Convolution2D(24, 5, 5, border_mode='valid', activation='elu'))
    model.add(Dropout(0.5))
    model.add(Convolution2D(36, 5, 5, border_mode='valid', activation='elu'))
    model.add(Dropout(0.5))
    model.add(Convolution2D(48, 5, 5, border_mode='valid', activation='elu'))
    model.add(Dropout(0.5))
    model.add(Convolution2D(64, 3, 3, border_mode='valid', activation='elu'))
    model.add(Dropout(0.5))
    model.add(Convolution2D(64, 3, 3, border_mode='valid', activation='elu'))
    model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(Dense(1164))
    model.add(Dense(100))
    model.add(Dense(50))
    model.add(Dense(10))
    model.add(Dense(1))

    # model.add(Convolution2D(16, 3, 3))
    # model.add(MaxPooling2D((4, 4)))
    # model.add(Dropout(0.5))
    # model.add(Activation('relu'))
    # model.add(Flatten())
    # model.add(Dense(40))
    # model.add(Activation('relu'))
    # model.add(Dense(1))
    # model.add(Activation('softmax'))

    # model.add(core.Flatten(input_shape=(row, col, ch)))
    # model.add(core.Dense())

    samples_per_epoch = SAMPLES_PER_EPOCH
    nb_val_samples = int(SAMPLES_PER_EPOCH * 0.2)
    # model.compile('adam', 'categorical_crossentropy', ['accuracy'])
    # model.compile('adam', 'mean_squared_error', ['mean_squared_error'])
    model.compile(loss='mse', optimizer='adam')
    history = model.fit_generator(
        train_data, samples_per_epoch=len(train_samples),
        validation_data=validation_data,
        nb_val_samples=len(validation_samples),
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
