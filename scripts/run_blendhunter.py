#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""SCRIPT NAME

This module contains methods for running SExtractor on testing data.

:Author: Samuel Farrens <samuel.farrens@cea.fr>

:Credits: Alice Lacan

"""

import os
import sys
import numpy as np
import plaidml.keras
from modopt.interface.errors import warn
from utils import load
from os.path import expanduser

plaidml.keras.install_backend()


class BHRunner:

    def __init__(self, in_path, out_path, sigma_values, n_noise_reals=5,
                 data_dir='bh_data', weights_dir='bh_weights',
                 weights_str='weights', preds_str='bh_preds', train=True):

        self.in_path = in_path
        self.out_path = out_path
        self.sigma_values = sigma_values.astype(int)
        self.noise_reals = list(range(1, n_noise_reals + 1))
        self.data_dir = data_dir
        self.weights_dir = weights_dir
        self.weights_str = weights_str
        self.preds_str = preds_str
        self.train = True
        self._call_bh()

    def _get_in_path(self, sigma, noise_real):

        return '{}/{}_{}_{}'.format(self.in_path, self.data_dir,
                                    sigma, noise_real)

    def _get_weights_path(self, sigma, noise_real):

        weights_path = ('{}/{}/{}_{}_{}'.format(self.out_path,
                        self.weights_dir, self.weights_str, sigma, noise_real))

        if not os.path.isdir(weights_path):
            os.mkdir(weights_path)

        return weights_path

    def _train_bh(self, bh, input_path):

        bh.train(input_path + '/BlendHunterData', get_features=True,
                 train_top=True, fine_tune=False)

    def _get_preds(self, bh, input_path):

        return bh.predict(input_path + '/BlendHunterData/test/test',
                          weights_type='top')

    def _save_preds(self, preds, sigma, noise_real):

        path = '../results/bh_results'
        np.save('{}/{}_{}_{}'.format(path, self.preds_str, sigma, noise_real),
                preds)

    def _call_bh(self):

        for sigma in self.sigma_values:
            for noise_real in self.noise_reals:

                input_path = self._get_in_path(sigma, noise_real)
                weights_path = self._get_weights_path(sigma, noise_real)

                bh = BlendHunter(weights_path=weights_path)

                if os.path.isdir(input_path):

                    if self.train:
                        self._train_bh(bh, input_path)

                    self._save_preds(self._get_preds(bh, input_path), sigma,
                                     noise_real)

                else:

                    warn('{} not found in {}.'.format(input_path,
                         self.in_path))


# to be removed
user_home = expanduser("~")
bh_path = user_home + '/Desktop/alice/BlendHunter'
sys.path.extend([bh_path])
from blendhunter import BlendHunter

# Set paths
input_path = user_home + '/Desktop/bh_data'
results_path = '../results'

# Set sigma values
sigma_values = load('{}/{}'.format(results_path, 'sigmas.npy'))

# Run BlendHunter
BHRunner(input_path, results_path, sigma_values)
