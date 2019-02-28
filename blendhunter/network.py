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
from keras.applications import VGG16
from keras.optimizers import Adam, SGD
from keras.callbacks import ModelCheckpoint
from keras.callbacks import EarlyStopping
from keras.callbacks import ReduceLROnPlateau


class BlendFindNet(object):

    def __init__(self, train_dir, valid_dir, train_labels=None,
                 valid_labels=None, image_shape=None, epochs_top=500,
                 epochs_fine=50, batch_size_top=256, batch_size_fine=16,
                 classes=('blended', 'not_blended'),
                 save_bottleneck=True, bottleneck_path='./',
                 top_model_file='./top_model_weights.h5',
                 final_model_file='./final_model_weights.h5'):

        self._epochs_top = epochs_top
        self._epochs_fine = epochs_fine
        self._batch_size_top = batch_size_top
        self._batch_size_fine = batch_size_fine
        self._classes = classes
        self._save_bottleneck = save_bottleneck
        self._bottleneck_path = bottleneck_path
        self._top_model_file = top_model_file
        self._final_model_file = final_model_file
        self._features = {'train': {}, 'valid': {}}
        self._features['train']['dir'] = train_dir
        self._features['valid']['dir'] = valid_dir
        self._features['train']['labels'] = train_labels
        self._features['valid']['labels'] = valid_labels

        if not isinstance(image_shape, type(None)):
            self._image_shape = image_shape
        else:
            self._get_image_shape()
        self._target_size = self._image_shape[:2]

    def _get_image_shape(self):

        path = '{}/{}'.format(self._features['train']['dir'],
                              self._classes[0])
        file = '{}/{}'.format(path, os.listdir(path)[0])

        self._image_shape = Image.open(file).size

    def _load_generator(self, input_dir, batch_size=None,
                        class_mode=None, augmentation=False):

        if augmentation:
            datagen = ImageDataGenerator(rescale=1. / 255,
                                         shear_range=0.2,
                                         zoom_range=0.2,
                                         horizontal_flip=True)
        else:
            datagen = ImageDataGenerator(rescale=1. / 255)

        generator = (datagen.flow_from_directory(input_dir,
                     target_size=self._target_size,
                     batch_size=batch_size, class_mode=class_mode,
                     shuffle=False))
        generator.steps = generator.n // generator.batch_size

        return generator

    def _get_bottleneck_feature(self, input_dir):

        generator = self._load_generator(input_dir,
                                         batch_size=self._batch_size_top)

        return (self._vgg16_model.predict_generator(generator,
                generator.steps))

    @staticmethod
    def _save_bottleneck_feature(bot_feat, data_type):

        file_name = 'bottleneck_features_{}.npy'.format(data_type)
        np.save(file_name, bot_feat)

    def _load_bottleneck_feature(self, data_type):

        file_name = ('{}bottleneck_features_{}.npy'.format(
                     self._bottleneck_path, data_type))
        if os.path.isfile(file_name):
            return np.load(file_name)
        else:
            raise IOError('{} not found'.format(file_name))

    @staticmethod
    def _build_vgg16_model(input_shape=None):

        return VGG16(include_top=False, weights='imagenet',
                     input_shape=input_shape)

    def _get_bottleneck_features(self):

        self._vgg16_model = self._build_vgg16_model()

        for key, value in self._features.items():

            bot_feat = self._get_bottleneck_feature(value['dir'])

            if self._save_bottleneck:
                self._save_bottleneck_feature(bot_feat, key)
                value['bottleneck'] = self._load_bottleneck_feature(key)
            else:
                value['bottleneck'] = bot_feat

    def _load_bottleneck_features(self):

        for key, value in self._features.items():
            if 'bottleneck' not in value:
                value['bottleneck'] = self._load_bottleneck_feature(key)

    def _set_labels(self):

        for key, value in self._features.items():
            if isinstance(value['labels'], type(None)):
                n_samp = value['bottleneck'].shape[0] // 2
                value['labels'] = np.array([0] * n_samp + [1] * n_samp)

    @staticmethod
    def _build_top_model(input_shape, dense_output=(256, 1024), dropout=0.1):

        model = Sequential()
        model.add(Flatten(input_shape=input_shape))
        model.add(Dense(dense_output[0]))
        model.add(Dropout(dropout))
        model.add(Dense(dense_output[1], activation='relu'))
        model.add(Dense(1, activation='sigmoid'))

        return model

    def _train_top_model(self):

        self._load_bottleneck_features()
        self._set_labels()

        model = (self._build_top_model(
                 input_shape=self._features['train']['bottleneck'].shape[1:]))

        model.compile(optimizer='adam', loss='binary_crossentropy',
                      metrics=['accuracy'])

        callbacks = []
        callbacks.append(ModelCheckpoint(self._top_model_file,
                         monitor='val_loss', verbose=1,
                         save_best_only=True, save_weights_only=True,
                         mode='auto', period=1))

        callbacks.append(EarlyStopping(monitor='val_loss', min_delta=0.001,
                                       patience=10, verbose=1))

        callbacks.append(ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                                           patience=5, min_delta=0.001,
                                           cooldown=2, verbose=1))

        model.fit(self._features['train']['bottleneck'],
                  self._features['train']['labels'],
                  epochs=self._epochs_top, batch_size=self._batch_size_top,
                  callbacks=callbacks,
                  validation_data=(self._features['valid']['bottleneck'],
                                   self._features['valid']['labels']))

        model.save_weights(self._top_model_file)

    def _freeze_layers(self, model, depth):

        for layer in model.layers[:depth]:
            layer.trainable = False

    def _fine_tune(self):

        vgg16_model = self._build_vgg16_model(self._image_shape)
        top_model = self._build_top_model(vgg16_model.output_shape[1:],
                                          dropout=0.4)
        top_model.load_weights(self._top_model_file)

        model = Model(inputs=vgg16_model.input,
                      outputs=top_model(vgg16_model.output))

        self._freeze_layers(model, 18)

        model.compile(loss='binary_crossentropy',
                      optimizer=Adam(lr=0.0001),
                      metrics=['binary_accuracy'])

        train_gen = self._load_generator(self._features['train']['dir'],
                                         batch_size=self._batch_size_fine,
                                         class_mode='binary',
                                         augmentation=True)

        valid_gen = self._load_generator(self._features['valid']['dir'],
                                         batch_size=self._batch_size_fine,
                                         class_mode='binary')

        callbacks = []
        callbacks.append(ModelCheckpoint('./vgg16_weights_best.h5',
                         monitor='val_loss', verbose=1,
                         save_best_only=True, save_weights_only=True,
                         mode='auto', period=1))
        callbacks.append(EarlyStopping(monitor='val_loss', min_delta=0.001,
                                       patience=10, verbose=1))
        callbacks.append(ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                                           patience=5, epsilon=0.001,
                                           cooldown=2, verbose=1))

        model.summary()

        model.fit_generator(train_gen, steps_per_epoch=train_gen.steps,
                            epochs=self._epochs_fine,
                            callbacks=callbacks,
                            validation_data=valid_gen,
                            validation_steps=valid_gen.steps)

        model.layers[19].trainable = False
        model.layers[17].trainable = True

        model.compile(loss='binary_crossentropy',
                      optimizer=SGD(lr=10e-5),
                      metrics=['binary_accuracy'])

        model.fit_generator(train_gen, steps_per_epoch=train_gen.steps,
                            epochs=self._epochs_fine,
                            callbacks=callbacks,
                            validation_data=valid_gen,
                            validation_steps=valid_gen.steps)

        model.save_weights(self._final_model_file)
