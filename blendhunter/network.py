# -*- coding: utf-8 -*-

""" NETWORK

:Author: Samuel Farrens <samuel.farrens@cea.fr>

"""

import os
import numpy as np
from cv2 import imread
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential, Model
from keras.layers import Dropout, Flatten, Dense, Input
from keras.applications import VGG16
from keras.optimizers import Adam, SGD
from keras.callbacks import ModelCheckpoint
from keras.callbacks import EarlyStopping
from keras.callbacks import ReduceLROnPlateau


class BlendHunter(object):
    """ BlendHunter

    Class for identifying blended galaxy images in postage stamps.

    Parameters
    ----------
    image_shape : tuple, optional
        Expected shape of input images
    classes : tuple, optional
        List of classes, default is ('blended', 'not_blended')
    final_model_file : str, optional
        File name of the final model weights, default is
        './final_model_weights'

    """

    def __init__(self, image_shape=None, classes=('blended', 'not_blended'),
                 final_model_file='./final_model_weights', verbose=1):

        self._image_shape = image_shape
        self._classes = classes
        self._final_model_file = final_model_file
        self._verbose = verbose

    def _get_image_shape(self, file):
        """ Get Image Shape

        Get the input image shape from an example image.

        Parameters
        ----------
        file : str
            File name

        """

        self._image_shape = imread(file).shape

    def _get_target_shape(self, image_path=None):
        """ Get Target Shape

        Get the network target shape from the image shape.

        Parameters
        ----------
        image_path : str, optional
            Path to image file

        """

        if isinstance(self._image_shape, type(None)) and image_path:
            file = '{}/{}'.format(image_path, os.listdir(image_path)[0])
            self._get_image_shape(file)

        self._target_size = self._image_shape[:2]

    def _load_generator(self, input_dir, batch_size=None,
                        class_mode=None, augmentation=False):
        """ Load Generator

        Load files from an input directory into a Keras generator.

        Parameters
        ----------
        input_dir : str
            Input directory
        batch_size : int, optional
            Batch size
        class_mode : str, optional
            Generator class mode
        shuffle : bool, optional
            Option to shuffle input files

        Returns
        -------
        keras_preprocessing.image.DirectoryIterator
            Keras generator

        """

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
        """ Get Bottleneck Feature

        Get bottleneck feature from VGG16 model.

        Parameters
        ----------
        input_dir : str
            Input directory

        Returns
        -------
        np.ndarray
            VGG16 bottleneck feature

        """

        generator = self._load_generator(input_dir,
                                         batch_size=self._batch_size_top)

        return (self._vgg16_model.predict_generator(generator,
                generator.steps))

    def _save_bottleneck_feature(self, bot_feat, data_type):
        """ Save Bottleneck Feature

        Save bottleneck feature to file.

        Parameters
        ----------
        bot_feat : np.ndarray
            Bottleneck feature
        data_type : str
            Type of feature to be saved

        """

        file_name = '{}_{}.npy'.format(self._bottleneck_file, data_type)
        np.save(file_name, bot_feat)

    def _load_bottleneck_feature(self, data_type):
        """ Load Bottleneck Feature

        Load bottleneck feature from file.

        Parameters
        ----------
        data_type : str
            Type of feature to be loaded

        """

        file_name = '{}_{}.npy'.format(self._bottleneck_file, data_type)
        if os.path.isfile(file_name):
            return np.load(file_name)
        else:
            raise IOError('{} not found'.format(file_name))

    @staticmethod
    def _build_vgg16_model(input_shape=None):
        """ Build VGG16 Model

        Build VGG16 CNN model using imagenet weights.

        Parameters
        ----------
        input_shape : str, optional
            Input data shape

        Returns
        -------

            VGG16 model

        """

        return VGG16(include_top=False, weights='imagenet',
                     input_shape=input_shape)

    def _get_bottleneck_features(self):
        """ Get Bottleneck Features

        Get the bottleneck features from the VGG16 model.

        """

        self._vgg16_model = self._build_vgg16_model()

        for key, value in self._features.items():

            bot_feat = self._get_bottleneck_feature(value['dir'])

            if self._save_bottleneck:
                self._save_bottleneck_feature(bot_feat, key)
                value['bottleneck'] = self._load_bottleneck_feature(key)
            else:
                value['bottleneck'] = bot_feat

    def _load_bottleneck_features(self):
        """ Load Bottleneck Features

        Load VGG16 bottleneck features.

        """

        for key, value in self._features.items():
            if 'bottleneck' not in value:
                value['bottleneck'] = self._load_bottleneck_feature(key)

    def _set_labels(self):
        """ Set Labels

        Set training labels for trainging data.

        """

        for key, value in self._features.items():
            if isinstance(value['labels'], type(None)):
                n_samp = value['bottleneck'].shape[0] // 2
                value['labels'] = np.array([0] * n_samp + [1] * n_samp)

    @staticmethod
    def _build_top_model(input_shape, dense_output=(256, 1024), dropout=0.1):
        """ Build Top Model

        Build the fully connected layer of the network.

        Parameters
        ----------
        input_shape : tuple
            Input data shape
        dense_output : tuple, optional
            Size of dense output layers, default is (256, 1024)
        dropout : float, optional
            Dropout rate, default is 0.1

        Returns
        -------

            Fully connected top model

        """

        model = Sequential()
        model.add(Flatten(input_shape=input_shape))
        model.add(Dense(dense_output[0]))
        model.add(Dropout(dropout))
        model.add(Dense(dense_output[1], activation='relu'))
        model.add(Dense(1, activation='sigmoid'))

        return model

    def _train_top_model(self):
        """ Train Top Model

        Train fully connected top model of the network.

        """

        self._load_bottleneck_features()
        self._set_labels()

        model = (self._build_top_model(
                 input_shape=self._features['train']['bottleneck'].shape[1:]))

        model.compile(optimizer='adam', loss='binary_crossentropy',
                      metrics=['accuracy'])

        top_model_file = '{}.h5'.format(self._top_model_file)

        callbacks = []
        callbacks.append(ModelCheckpoint(top_model_file,
                         monitor='val_loss', verbose=self._verbose,
                         save_best_only=True, save_weights_only=True,
                         mode='auto', period=1))

        callbacks.append(EarlyStopping(monitor='val_loss', min_delta=0.001,
                                       patience=10, verbose=self._verbose))

        callbacks.append(ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                                           patience=5, min_delta=0.001,
                                           cooldown=2, verbose=self._verbose))

        model.fit(self._features['train']['bottleneck'],
                  self._features['train']['labels'],
                  epochs=self._epochs_top, batch_size=self._batch_size_top,
                  callbacks=callbacks,
                  validation_data=(self._features['valid']['bottleneck'],
                                   self._features['valid']['labels']))

        model.save_weights(top_model_file)

    def _freeze_layers(self, model, depth):
        """ Freeze Network Layers

        Parameters
        ----------
        model :
            Keras model
        depth : int
            Depth of layers to be frozen

        """

        for layer in model.layers[:depth]:
            layer.trainable = False

    def _build_final_model(self, load_top_weights=False,
                           load_final_weights=False):
        """ Build Final Model

        Build the final BlendHunter model.

        Parameters
        ----------
        load_top_weights : bool
            Option to load the top model weights
        load_final_weights : bool
            Option to load the final model weights

        Returns
        -------

            Final model

        """

        vgg16_model = self._build_vgg16_model(self._image_shape)
        top_model = self._build_top_model(vgg16_model.output_shape[1:],
                                          dropout=0.4)

        if load_top_weights:
            top_model.load_weights('{}.h5'.format(self._top_model_file))

        model = Model(inputs=vgg16_model.input,
                      outputs=top_model(vgg16_model.output))

        if load_final_weights:
            model.load_weights('{}.h5'.format(self._final_model_file))

        return model

    def _fine_tune(self):
        """ Fine Tune

        Fine tune the final model training.

        """

        model = self._build_final_model(load_top_weights=True)

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
                         monitor='val_loss', verbose=self._verbose,
                         save_best_only=True, save_weights_only=True,
                         mode='auto', period=1))
        callbacks.append(EarlyStopping(monitor='val_loss', min_delta=0.001,
                                       patience=10, verbose=self._verbose))
        callbacks.append(ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                                           patience=5, epsilon=0.001,
                                           cooldown=2, verbose=self._verbose))

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

        model.save_weights('{}.h5'.format(self._final_model_file))

    def train(self, train_dir, valid_dir, train_labels=None, valid_labels=None,
              epochs_top=500, epochs_fine=50, batch_size_top=256,
              batch_size_fine=16, save_bottleneck=True,
              bottleneck_file='./bottleneck_features',
              top_model_file='./top_model_weights',):
        """ Train

        Train the BlendHunter network.

        Parameters
        ----------
        train_dir : str
            Path to training data
        valid_dir : str
            Path to validation data
        train_labels : list
            Training data labels
        valid_labels : list
            Validation data labels
        epochs_top : int, optional
            Number of training epochs for top model, default is 500
        epochs_fine : int, optional
            Number of training epochs for fine tuning, default is 50
        batch_size_top : int, optional
            Batch size for top model, default is 256
        batch_size_fine : int, optional
            Batch size for fine tuning, default is 16
        save_bottleneck : bool, optional
            Option to save bottleneck features, default is True
        bottleneck_file : str, optional
            File name for bottleneck features, default is
            './bottleneck_features'
        top_model_file : str, optional
            File name for top model weights, default is './top_model_weights'

        """

        self._epochs_top = epochs_top
        self._epochs_fine = epochs_fine
        self._batch_size_top = batch_size_top
        self._batch_size_fine = batch_size_fine
        self._save_bottleneck = save_bottleneck
        self._bottleneck_file = bottleneck_file
        self._top_model_file = top_model_file
        self._features = {'train': {}, 'valid': {}}
        self._features['train']['dir'] = train_dir
        self._features['valid']['dir'] = valid_dir
        self._features['train']['labels'] = train_labels
        self._features['valid']['labels'] = valid_labels

        self._get_target_shape('{}/{}'.format(self._features['train']['dir'],
                               self._classes[0]))
        self._get_bottleneck_features()
        self._train_top_model()
        self._fine_tune()

    def predict(self, input_path=None, input_path_keras=None, input_data=None):
        """ Predict

        Predict classes for test data

        Parameters
        ----------
        input_path : str
            Path to input data
        input_path_keras : str
            Path to input data in Keras format, i.e. path to directory one
            level above where the data is stored
        input_data : np.ndarray
            Array of input images

        Returns
        -------
        dict
            Dictionary of file names and corresponding classes

        """

        if input_path:
            test_path = '/'.join(input_path.split('/')[:-1])
        elif input_path_keras:
            test_path = input_path_keras
        else:
            test_path = None

        if test_path:

            self._get_target_shape('{}/{}'.format(test_path,
                                   os.listdir(test_path)[0]))
            model = self._build_final_model(load_final_weights=True)
            test_gen = self._load_generator(test_path, batch_size=1)
            test_gen.reset()
            res = model.predict_generator(test_gen,
                                          verbose=self._verbose).flatten()

        elif not isinstance(input_data, type(None)):

            self._image_shape = input_data.shape[1:]
            self._get_target_shape()
            model = self._build_final_model(load_final_weights=True)
            res = model.predict(input_data, verbose=self._verbose).flatten()

        else:

            raise RuntimeError('No input data provided.')

        labels = {0: self._classes[0], 1: self._classes[1]}
        preds = [labels[k] for k in np.around(res)]

        return preds
