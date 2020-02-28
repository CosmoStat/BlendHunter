#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""PREPARE DATA

This module contains methods for preparing the training, validation and testing
data.

:Author: Samuel Farrens <samuel.farrens@cea.fr>

:Credits: Alice Lacan

"""

import os
import numpy as np
from modopt.signal.noise import add_noise
from modopt.base.np_adjust import pad2d
from utils import load

# to be removed
import sys
from os.path import expanduser
user_home = expanduser("~")
bh_path = user_home + '/Desktop/alice/BlendHunter'
sys.path.extend([bh_path])
from blendhunter.data import CreateTrainData


class PrepData:

    def __init__(self, in_path, out_path, sigma_values, n_noise_reals=5,
                 padding=(7, 7), output_str='bh_data',
                 bh_data_dir='BlendHunterData', sep_data_dir='sepData'):

        self.in_path = in_path
        self.out_path = out_path
        self.sigma_values = sigma_values
        self.noise_reals = list(range(1, n_noise_reals + 1))
        self.padding = padding
        self.output_str = output_str
        self.bh_data_dir = bh_data_dir
        self.sep_data_dir = sep_data_dir
        self._prep_data()

    def _load_mocks(self):

        blended = load(self.in_path + '/blended/gal_obj_0.npy')
        not_blended = load(self.in_path + '/not_blended/gal_obj_0.npy')

        return np.array([blended, not_blended])

    @staticmethod
    def _check_dir(dir):

        return os.path.isdir(dir)

    @classmethod
    def _create_dir(cls, dir):

        if not cls._check_dir(dir):
            os.mkdir(dir)

    def _get_output_dir(self, sigma, noise_real):

        output_dir = '{}/{}_{}_{}'.format(self.out_path, self.output_str,
                                          int(sigma), noise_real)

        self._create_dir(output_dir)

        return output_dir

    def _pad_noise(self, image, sigma):

        return add_noise(pad2d(image, self.padding), sigma=sigma)

    def _get_pad_sample(self, samples, sigma):

        padded_images = []

        for sample in samples:
            for image in sample:
                im_pad = self._pad_noise(image['galsim_image'][0].array, sigma)
                image['galsim_image_noisy'] = im_pad
                padded_images.append(im_pad)

        return np.array(padded_images).reshape(samples.shape + im_pad.shape)

        # return np.array([[self._pad_noise(image['galsim_image'][0].array,
        #                   sigma) for image in sample]
        #                  for sample in samples])

    def _prep_train_data(self, data, output_dir):

        if not self._check_dir('{}/{}'.format(output_dir, self.bh_data_dir)):

            ctd = CreateTrainData(data, output_dir)
            ctd.prep_axel(path_to_output=output_dir)

    def _prep_sep_data(self, data, output_dir):

        sep_output_dir = '{}/{}'.format(output_dir, self.sep_data_dir)

        self._create_dir(sep_output_dir)

        indices = slice(36000, 40000)

        np.save('{}/blended.npy'.format(sep_output_dir), data[0][indices])
        np.save('{}/not_blended.npy'.format(sep_output_dir), data[1][indices])

    def _prep_data(self):

        samples = self._load_mocks()

        for sigma in self.sigma_values:
            for noise_real in self.noise_reals:

                output_dir = self._get_output_dir(sigma, noise_real)
                samples_pad = self._get_pad_sample(samples, sigma)
                self._prep_train_data(samples_pad, output_dir)
                self._prep_sep_data(samples, output_dir)


# Set paths
input_path = ('/Users/Shared/axel_sims/larger_dataset')
output_path = user_home + '/Desktop/bh_data'

# Set sigma values
results_path = '../results'
sigma_values = np.array([5.0, 14.0, 18.0, 26.0, 35.0, 40.0])
np.save('{}/sigmas.npy'.format(results_path), sigma_values)

# Prepare the dataset
PrepData(input_path, output_path, sigma_values)
