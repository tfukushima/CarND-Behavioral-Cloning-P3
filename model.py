#!/bin/env python

"""Self-driving car nanodegree on Udacity; Project 3: Behaviour cloning
"""

import csv
import os

from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.layers.core import Activation, Dense, Dropout, Flatten
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


COURSES=['c1', 'c2']

CSV_FILENAME = 'driving_log.csv'
IMG_DIRNAME = 'IMG'
CORRECTION = 0.2
BATCH_SIZE = 32
EPOCH = 15


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


def load_data(dir_path, samples, batch_size=BATCH_SIZE):
    """Load the data from CSV and images files into the memory.

    This function is the generator to yield the loaded image data in the batch
    size.

    :param dir_path:   The path to the directory that contains the CSV file and
                       the directory has image files to load.
    :param samples:    The loaded steering data sample as numpy array.
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
        
                image_center_name = batch_sample[0].split('/')[-1]
                # image_center = cv2.imread(os.path.join(
                image_center = Image.open(os.path.join(
                    dir_path, IMG_DIRNAME, image_center_name))
                image_left_name = batch_sample[1].split('/')[-1]
        
                images.append(image_center)
                angles.append(steering_center)
        
                # Flip image and the angle
                image_center_flipped = np.fliplr(image_center)
                images.append(image_center_flipped)
                steering_center_flipped = - steering_center
                angles.append(steering_center_flipped)
                
                if image_left_name:
                    # image_left = cv2.imread(os.path.join(
                    image_left = Image.open(os.path.join(
                        dir_path, IMG_DIRNAME, image_left_name))
                    images.append(image_left)
                    images.append(steering_left)
                    
                    image_left_flipped = np.fliplr(image_left)
                    images.append(image_left_flipped)
                    steering_left_flipped = - steering_left
                    angles.append(steering_left_flipped)
        
                image_right_name = batch_sample[2].split('/')[-1]
                if image_right_name:
                    # image_right = cv2.imread(os.path.join(
                    image_right = Image.open(os.path.join(
                        dir_path, IMG_DIRNAME, image_right_name))
                    images.append(image_right)
                    images.append(steering_right)
        
                    image_right_flipped = np.fliplr(image_right)
                    images.append(image_right_flipped)
                    steering_right_flipped = - steering_right
                    angles.append(steering_right_flipped)
        
                X_train = np.array(images)
                y_train = np.array(angles)
                
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
    for course in COURSES:
        samples = load_steering_samples(course)
        train_samples, validation_samples = model_selection.train_test_split(
            samples, test_size=0.2)

        train_data = load_data(course, train_samples)
        validation_data = load_data(course, validation_samples)

        
        model = Sequential()
        # Crop the image data
        model.add(Cropping2D(cropping=((50, 30), (0, 0)), input_shape=(3, 160, 320)))
        # Normalize the image data
        ch, row, col = 3, 80, 320
        model.add(Lambda(lambda x: x/127.5 - 1.0,
                         input_shape=(ch, row, col),
                         output_shape=(ch, row, col)))
        # Convolutional layer
        model.add(Convolution2D(80, 4, 4))
        model.add(MaxPooling2D((2, 2)))
        model.add(Dropout(0.5))
        model.add(Activation('relu'))
        model.add(Flatten())
        model.add(Dense(40))
        model.add(Activation('relu'))
        model.add(Dense(1))
        model.add(Activation('softmax'))

        # model.add(core.Flatten(input_shape=(row, col, ch)))
        # model.add(core.Dense())

        model.compile('adam', 'categorical_crossentropy', ['accuracy'])
        history = model.fit_generator(
            train_data, samples_per_epoch=len(train_data),
            validation_data=validation_data,
            nb_val_samples=len(validation_data),
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
