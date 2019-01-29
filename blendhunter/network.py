# -*- coding: utf-8 -*-

""" NETWORK

:Author: Samuel Farrens <samuel.farrens@cea.fr>

"""

import os
import numpy as np
from astropy.io import fits
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential, Model
from keras.layers import Dropout, Flatten, Dense, Input
from keras import applications
from keras import optimizers
from keras.callbacks import ModelCheckpoint
from keras.callbacks import EarlyStopping
from keras.callbacks import ReduceLROnPlateau


class VGG16(object):

    def __init__(self, train_dir, valid_dir, image_shape=None, epochs=500,
                 batch_size=256):

        self._input_dirs = (train_dir, valid_dir)
        self._epochs = epochs
        self._batch_size = batch_size

        if not isinstance(image_shape, type(None)):
            self._image_shape = image_shape
        else:
            self._get_image_shape()

        self._get_n_samples()

    def _get_image_shape(self):

        self._image_size = fits.getdata('{}/{}'.format(self._input_dirs[0],
                                        os.listdir(
                                        self._input_dirs[0])[0])).shape

    def _get_n_samples(self):

        file_list = [[file for file in os.listdir(dir) if
                     os.path.isfile(file)] for dir in self._input_dirs]

        self._n_samples = [len(sublist) for sublist in file_list]

    def train_network(self):

        self._save_bottleneck_features()

    def _save_bottleneck_feature(self, input_dir, output_file, n_samples):

        generator = (datagen.flow_from_directory(input_dir,
                     target_size=(self._image_shape),
                     batch_size=self._batch_size, class_mode=None,
                     shuffle=False))

        bottleneck_features = (model.predict_generator(generator,
                               n_samples // self._batch_size))

        np.save(open(output_file, 'wb'), bottleneck_features)

    def _save_bottleneck_features(self):

        datagen = ImageDataGenerator(rescale=1. / 255)
        model = applications.VGG16(include_top=False, weights='imagenet')

        output_files = ('bottleneck_features_train.npy',
                        'bottleneck_features_valid.npy')

        for input, output, n_samp in zip(self._input_dirs, output_files,
                                         self._n_samples):

            self._save_bottleneck_feature(input, output, n_samp)
