# -*- coding: utf-8 -*-

""" NETWORK

This module defines the BlendHunter class which can be used to retrain the
network or use predefined weights to make predictions on unseen data.

:Author: Samuel Farrens <samuel.farrens@cea.fr>

"""

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from time import time
from cv2 import imread
import keras
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
    weights_path : str, optional
        Path to weights, default is './weights'
    top_model_file : str, optional
        File name for top model weights, default is 'top_model_weights'
    final_model_file : str, optional
        File name of the final model weights, default is
        'final_model_weights'

    """

    def __init__(self, image_shape=None, classes=('blended', 'not_blended'),
                 weights_path='./weights', top_model_file='top_model_weights',
                 final_model_file='final_model_weights', verbose=0):

        self._image_shape = image_shape
        self._classes = classes
        self._weights_path = weights_path
        self._top_model_file = self._format(weights_path, top_model_file)
        self._final_model_file = self._format(weights_path, final_model_file)
        self._verbose = verbose
        self.history = None

    @staticmethod
    def _format(path, name):
        """ Format

        Add path to name.

        Parameters
        ----------
        path : str
            Base path
        name : str
            Path extension

        Returns
        -------
        str
            Formated path

        """

        return '{}/{}'.format(path, name)

    def getkwarg(self, key, default=None):
        """ Get keyword agrument

        Get value from keyword agruments if it exists otherwise return default.

        Parameters
        ----------
        key : str
            Dictionary key
        default : optional
            Default value

        """

        return self._kwargs[key] if key in self._kwargs else default

    @staticmethod
    def _get_image_shape(file):
        """ Get Image Shape

        Get the input image shape from an example image.

        Parameters
        ----------
        file : str
            File name

        Returns
        -------
        tuple
            Image shape

        """

        return imread(file).shape

    def _get_target_shape(self, image_path=None):
        """ Get Target Shape

        Get the network target shape from the image shape.

        Parameters
        ----------
        image_path : str, optional
            Path to image file

        """

        if isinstance(self._image_shape, type(None)) and image_path:
            file = self._format(image_path, os.listdir(image_path)[0])
            self._image_shape = self._get_image_shape(file)

        self._target_size = self._image_shape[:2]

    def _load_generator(self, input_dir, batch_size=None,
                        class_mode=None, augmentation=True):
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

    def _get_feature(self, input_dir):
        """ Get Feature

        Get network feature and labels from VGG16 model.

        Parameters
        ----------
        input_dir : str
            Input directory

        Returns
        -------
        tuple
            VGG16 bottleneck feature, class labels

        """

        generator = self._load_generator(input_dir,
                                         batch_size=self._batch_size_top)
        labels = generator.classes[:generator.steps * self._batch_size_top]

        return (self._vgg16_model.predict_generator(generator,
                generator.steps), labels)

    @staticmethod
    def _save_data(data, data_type, file_path):
        """ Save Data

        Save data to file.

        Parameters
        ----------
        data : np.ndarray
            Output data
        data_type : str
            Type of feature to be saved
        file_path : str
            File path

        """

        file_name = '{}_{}.npy'.format(file_path, data_type)
        np.save(file_name, data)

    @staticmethod
    def _load_data(data_type, file_path):
        """ Load Data

        Load data from file.

        Parameters
        ----------
        data_type : str
            Type of feature to be loaded
        file_path : str
            File path

        """

        file_name = '{}_{}.npy'.format(file_path, data_type)
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

    def _get_features(self):
        """ Get Features

        Get the network (bottleneck) features from the VGG16 model.

        """

        self._vgg16_model = self._build_vgg16_model()

        for key, value in self._features.items():

            bot_feat, labels = self._get_feature(value['dir'])

            if self._save_bottleneck:
                self._save_data(bot_feat, key, self._bottleneck_file)

            if self._save_labels:
                self._save_data(labels, key, self._labels_file)

            value['bottleneck'] = bot_feat
            value['labels'] = labels

    def _load_features(self):
        """ Load Bottleneck Features

        Load VGG16 bottleneck features.

        """

        for feature_name in ('bottleneck', 'labels'):

            if feature_name == 'bottleneck':
                out_path = self._bottleneck_file
            else:
                out_path = self._labels_file

            for key, value in self._features.items():
                if feature_name not in value:
                    value[feature_name] = self._load_data(key, out_path)

    @staticmethod
    def _build_top_model(input_shape, dense_output=(256, 1024), dropout=0.1):
        """ Build Top Model

        Build the fully connected layers of the network.

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
        keras.model
            Fully connected top model

        """

        model = Sequential()
        model.add(Flatten(input_shape=input_shape))
        model.add(Dense(dense_output[0], kernel_initializer=keras.initializers.Constant(value=0)))
        model.add(Dropout(dropout))
        model.add(Dense(dense_output[1], activation='relu', kernel_initializer=keras.initializers.RandomNormal(seed=12345)))
        model.add(Dense(1, activation='sigmoid', kernel_initializer=keras.initializers.RandomNormal(seed=12345)))

        return model

    def _train_top_model(self):
        """ Train Top Model

        Train fully connected top model of the network.

        """

        self._load_features()

        model = (self._build_top_model(
                 input_shape=self._features['train']['bottleneck'].shape[1:]))

        model.compile(optimizer=self.getkwarg('top_opt', 'adam'),
                      loss=self.getkwarg('top_loss', 'binary_crossentropy'),
                      metrics=self.getkwarg('top_metrics', ['accuracy']))

        top_model_file = '{}.h5'.format(self._top_model_file)

        callbacks = []
        callbacks.append(ModelCheckpoint(top_model_file,
                         monitor='val_loss', verbose=self._verbose,
                         save_best_only=True, save_weights_only=True,
                         mode='auto', period=1))

        if self.getkwarg('top_early_stop', True):

            min_delta = self.getkwarg('top_min_delta', 0.001)
            patience = self.getkwarg('top_patience', 10)

            callbacks.append(EarlyStopping(monitor='val_loss',
                                           min_delta=min_delta,
                                           patience=patience,
                                           verbose=self._verbose))

        callbacks.append(ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                                           patience=5, min_delta=0.001,
                                           cooldown=2, verbose=self._verbose))

        self.history = (model.fit(self._features['train']['bottleneck'],
                        self._features['train']['labels'],
                        epochs=self._epochs_top,
                        batch_size=self._batch_size_top,
                        callbacks=callbacks,
                        validation_data=(self._features['valid']['bottleneck'],
                                         self._features['valid']['labels']),
                        verbose=self._verbose))

        model.save_weights(top_model_file)

    def plot_history(self):
        """ Plot History

        Plot the training history metrics.

        """

        sns.set(style="darkgrid")

        if not isinstance(self.history, type(None)):

            plt.figure(figsize=(16, 8))

            plt.subplot(121)
            plt.plot(self.history.history['acc'])
            plt.plot(self.history.history['val_acc'])
            plt.title('Model Accuracy')
            plt.ylabel('Accuracy')
            plt.xlabel('Epoch')
            plt.legend(['train', 'valid'], loc='upper left')

            plt.subplot(122)
            plt.plot(self.history.history['loss'])
            plt.plot(self.history.history['val_loss'])
            plt.title('Model Loss')
            plt.ylabel('Loss')
            plt.xlabel('Epoch')
            plt.legend(['train', 'valid'], loc='upper left')

            plt.show()

        else:

            print('No history to display. Run training first.')

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
                                          dropout=0.1)

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
        callbacks.append(ModelCheckpoint('{}.h5'.format(self._fine_tune_file),
                         monitor='val_loss', verbose=self._verbose,
                         save_best_only=True, save_weights_only=True,
                         mode='auto', period=1))
        callbacks.append(EarlyStopping(monitor='val_loss', min_delta=0.001,
                                       patience=10, verbose=self._verbose))
        callbacks.append(ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                                           patience=5, min_delta=0.001,
                                           cooldown=2, verbose=self._verbose))

        model.fit_generator(train_gen, steps_per_epoch=train_gen.steps,
                            epochs=self._epochs_fine,
                            callbacks=callbacks,
                            validation_data=valid_gen,
                            validation_steps=valid_gen.steps,
                            verbose=self._verbose)

        self._freeze_layers(model, 19)
        model.layers[17].trainable = True

        model.compile(loss='binary_crossentropy',
                      optimizer=SGD(lr=10e-5),
                      metrics=['binary_accuracy'])

        model.fit_generator(train_gen, steps_per_epoch=train_gen.steps,
                            epochs=self._epochs_fine,
                            callbacks=callbacks,
                            validation_data=valid_gen,
                            validation_steps=valid_gen.steps,
                            verbose=self._verbose)

        model.save_weights('{}.h5'.format(self._final_model_file))

    def train(self, input_path, get_features=True, train_top=True,
              fine_tune=True, train_dir_name='train',
              valid_dir_name='validation', epochs_top=500, epochs_fine=50,
              batch_size_top=250, batch_size_fine=16, save_bottleneck=True,
              bottleneck_file='bottleneck_features',
              save_labels=True, labels_file='labels',
              fine_tune_file='fine_tune_checkpoint',
              top_model_file='top_model_weights', **kwargs):
        """ Train

        Train the BlendHunter network.

        Parameters
        ----------
        input_path : str
            Path to input data
        get_features : bool, optional
            Option to get bottleneck features, default is True
        train_top : bool, optional
            Option to train top model, default is True
        fine_tune : bool, optional
            Option to run fine tuning component of training, default is True
        train_dir_name : str, optional
            Training data directory name, default is 'train'
        valid_dir_name : str, optional
            Validation data directory name, default is 'validation'
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
            'bottleneck_features'
        fine_tune_file : str, optional
            Training checkpoint for the fine tuning step, default is
            'fine_tune_checkpoint'

        """

        start = time()

        self._epochs_top = epochs_top
        self._epochs_fine = epochs_fine
        self._batch_size_top = batch_size_top
        self._batch_size_fine = batch_size_fine
        self._save_bottleneck = save_bottleneck
        self._save_labels = save_labels
        self._bottleneck_file = self._format(self._weights_path,
                                             bottleneck_file)
        self._labels_file = self._format(self._weights_path, labels_file)
        self._fine_tune_file = self._format(self._weights_path, fine_tune_file)
        self._features = {'train': {}, 'valid': {}}
        self._features['train']['dir'] = self._format(input_path,
                                                      train_dir_name)
        self._features['valid']['dir'] = self._format(input_path,
                                                      valid_dir_name)
        self._kwargs = kwargs

        self._get_target_shape(self._format(self._features['train']['dir'],
                               self._classes[0]))
        if get_features:
            self._get_features()
        if train_top:
            self._train_top_model()
        if fine_tune:
            self._fine_tune()

        end = time()

        print('Duration {:0.2f}s'.format(end - start))

    def predict(self, input_path=None, input_path_keras=None, input_data=None,
                weights_type='fine'):
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
        weights_type : str, optional {'fine', 'top'}
            Type of weights to use for predition, default is 'fine'

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
        if weights_type not in ('fine', 'top'):
            raise ValueError('Invalid value for weights_type. Options are '
                             '"fine" or "top"')

        if test_path:

            self._get_target_shape(self._format(test_path,
                                   os.listdir(test_path)[0]))
            if weights_type == 'fine':
                model = self._build_final_model(load_final_weights=True)
            elif weights_type == 'top':
                model = self._build_final_model(load_top_weights=True)
            test_gen = self._load_generator(test_path,
                                            class_mode='categorical',
                                            batch_size=1)
            self.filenames = test_gen.filenames
            test_gen.reset()

            res = model.predict_generator(test_gen,
                                          verbose=self._verbose,
                                          steps=test_gen.steps).flatten()


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
