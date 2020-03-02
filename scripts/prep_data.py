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

    def __init__(self, in_path, out_path, sigma_values=None, n_noise_reals=5,
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

        if sigma_values is None:
            self._prep_real_data()
        else:
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
            return True

        else:
            return False

    def _get_output_dir(self, sigma, noise_real):

        output_dir = '{}/{}_{}_{}'.format(self.out_path, self.output_str,
                                          int(sigma), noise_real)

        return output_dir

    def _pad_noise(self, image, sigma):

        return add_noise(pad2d(image, self.padding), sigma=sigma)

    def _get_pad_sample(self, samples, sigma=None):

        padded_images = []

        for sample in samples:
            for image in sample:
                if sigma is None:
                    im_pad = pad2d(image['galsim_image'][0].array,
                                   self.padding)
                else:
                    im_pad = self._pad_noise(image['galsim_image'][0].array,
                                             sigma)
                image['galsim_image_noisy'] = im_pad
                padded_images.append(im_pad)

        return np.array(padded_images).reshape(samples.shape + im_pad.shape)

    def _prep_train_data(self, data, output_dir, divide=True):

        train_fractions = (0.45, 0.45, .1) if divide else (0.0, 0.0, 1.0)

        if not self._check_dir('{}/{}'.format(output_dir, self.bh_data_dir)):

            ctd = CreateTrainData(data, output_dir, train_fractions)
            ctd.prep_axel(path_to_output=output_dir)

    def _prep_sep_data(self, data, output_dir, slice=True):

        sep_output_dir = '{}/{}'.format(output_dir, self.sep_data_dir)

        self._create_dir(sep_output_dir)

        if slice:
            indices = slice(36000, 40000)
            blends = data[0][indices]
            no_blends = data[1][indices]
        else:
            blends = data[0]
            no_blends = data[1]

        np.save('{}/blended.npy'.format(sep_output_dir), blends)
        np.save('{}/not_blended.npy'.format(sep_output_dir), no_blends)

    def _prep_sim_data(self):

        samples = self._load_mocks()

        for sigma in self.sigma_values:
            for noise_real in self.noise_reals:

                output_dir = self._get_output_dir(sigma, noise_real)

                if self._create_dir(output_dir):
                    samples_pad = self._get_pad_sample(samples, sigma=sigma)
                    self._prep_train_data(samples_pad, output_dir)
                    self._prep_sep_data(samples, output_dir)

    def _prep_real_data(self):

        output_dir = '{}/{}'.format(self.out_path, self.output_str)

        if self._create_dir(output_dir):
            samples = self._load_mocks()
            samples_pad = self._get_pad_sample(samples)
            self._prep_train_data(samples_pad, output_dir, divide=False)
            self._prep_sep_data(samples, output_dir, slice=False)


# Set paths
input_path_sim = ('/Users/Shared/axel_sims/larger_dataset')
output_path_sim = user_home + '/Desktop/bh_data'
input_path_cosmos = ('/Users/Shared/axel_sims/deblending_real/sample_10k')
output_path_cosmos = user_home + '/Desktop/cosmos_data'

# Set sigma values
results_path = '../results'
sigma_values = np.array([5.0, 14.0, 18.0, 26.0, 35.0, 40.0])
np.save('{}/sigmas.npy'.format(results_path), sigma_values)

# Prepare the simulated dataset
# PrepData(input_path_sim, output_path_sim, sigma_values)

# Prepare the COSMOS dataset
PrepData(input_path_cosmos, output_path_cosmos)
