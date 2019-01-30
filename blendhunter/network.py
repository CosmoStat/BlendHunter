# -*- coding: utf-8 -*-

""" NETWORK

:Author: Samuel Farrens <samuel.farrens@cea.fr>

"""

import os
import numpy as np
from PIL import Image
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
                 batch_size=256, classes=('blended', 'not_blended')):

        self._input_dirs = (train_dir, valid_dir)
        self._epochs = epochs
        self._batch_size = batch_size
        self._classes = classes
        self._save_bottleneck = False
        self._bot_feat = {}

        if not isinstance(image_shape, type(None)):
            self._image_shape = image_shape
        else:
            self._get_image_shape()

        self._get_n_samples()

    def _get_image_shape(self):

        path = '{}/{}'.format(self._input_dirs[0], self._classes[0])
        file = '{}/{}'.format(path, os.listdir(path)[0])

        self._image_shape = Image.open(file).size

    def _get_n_samples(self):

        file_list = [[os.listdir('{}/{}'.format(dir, _class)) for _class in
                     self._classes] for dir in self._input_dirs]

        n_items = [len(item) for sublist in file_list for item in sublist]

        self._train_labels = [0] * n_items[0] + [1] * n_items[1]
        self._valid_labels = [0] * n_items[2] + [1] * n_items[3]

        self._n_samples = (n_items[0] + n_items[1], n_items[2] + n_items[3])

    def train_network(self):

        self._get_bottleneck_features()
        self._train_top_model()

    def _get_bottleneck_feature(self, input_dir, n_samples):

        generator = (self._datagen.flow_from_directory(input_dir,
                     target_size=(self._image_shape),
                     batch_size=self._batch_size, class_mode=None,
                     shuffle=False))

        return (self._model.predict_generator(generator,
                n_samples // self._batch_size))

    def _get_bottleneck_features(self):

        self._datagen = ImageDataGenerator(rescale=1. / 255)
        self._model = applications.VGG16(include_top=False,
                                         weights='imagenet')

        data_types = ('train', 'valid')

        for input, n_samp, data_type in zip(self._input_dirs, self._n_samples,
                                            data_types):

            bot_feat = self._get_bottleneck_feature(input, n_samp)

            if self._save_bottleneck:
                file_name = 'bottleneck_features_{}.npy'.fortmat(data_type)
                np.save(file_name, bot_feat)
            else:
                self._bot_feat[data_type] = bot_feat

    def _train_top_model(self):

        if os.path.isfile('bottleneck_features_train.npy'):
            self._bot_feat['train'] = np.load('bottleneck_features_train.npy')

        if os.path.isfile('bottleneck_features_valid.npy'):
            self._bot_feat['valid'] = np.load('bottleneck_features_valid.npy')

        top_model_weights_path = 'bottleneck_fc_model.h5'

        model = Sequential()
        model.add(Flatten(input_shape=self._bot_feat['train'].shape[1:]))
        model.add(Dense(256))
        model.add(Dropout(0.1))
        model.add(Dense(1024, activation='relu'))
        model.add(Dense(1, activation='sigmoid'))

        model.compile(optimizer='adam', loss='binary_crossentropy',
                      metrics=['accuracy'])

        callbacks = []
        callbacks.append(ModelCheckpoint(top_model_weights_path,
                         monitor='val_loss', verbose=1,
                         save_best_only=True, save_weights_only=True,
                         mode='auto', period=1))

        callbacks.append(EarlyStopping(monitor='val_loss', min_delta=0.001,
                                       patience=10, verbose=1))

        callbacks.append(ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                                           patience=5, min_delta=0.001,
                                           cooldown=2, verbose=1))

        model.fit(self._bot_feat['train'], self._train_labels,
                  epochs=self._epochs, batch_size=self._batch_size,
                  callbacks=callbacks,
                  validation_data=(self._bot_feat['valid'],
                                   self._valid_labels))
        model.save_weights(top_model_weights_path)
