#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""SCRIPT NAME

This module contains methods for running SExtractor on testing data.

:Author: Samuel Farrens <samuel.farrens@cea.fr>

:Credits: Alice Lacan

"""

import numpy as np
from utils import load, DataHandler
from sep_script import Run_Sep
from os.path import expanduser


class CallSepRunner:

    def __init__(self, path, sigma_values, n_noise_reals=5, prefix='bh_pad',
                 output_str='sep_preds'):

        self.path = path
        self.sigma_values = sigma_values
        self.noise_reals = [''] + list(range(1, n_noise_reals))
        self.prefix = prefix
        self.output_str = output_str
        self._call_sep_run()

    def _load_data(self, sigma, noise_real):

        return DataHandler('{}/{}{}{}'.format(self.path, self.prefix,
                           sigma, noise_real)).datasets

    def _run_sep(self, blended, not_blended):

        runner = Run_Sep()

        preds_b, _ = runner.process(blended)
        preds_nb, _ = runner.process(not_blended)

        return np.concatenate((preds_b, preds_nb), axis=0)

    def _save_preds(self, preds, sigma, noise_real):

        # to be removed
        path = '../results/sep_results'
        np.save('{}/{}_{}_{}'.format(path, self.output_str, sigma, noise_real),
                preds)

        # np.save(preds, '{}/{}_{}_{}'.format(self.path, self.output_str, sigma,
        #         noise_real))

    def _call_sep_run(self):

        for sigma in self.sigma_values:
            for noise_real in self.noise_reals:

                preds = self._run_sep(*self._load_data(int(sigma), noise_real))
                self._save_preds(preds, int(sigma), noise_real)


# to be removed
user_home = expanduser("~")
path = user_home + '/Desktop/alice/results'

# Set output path
results_path = '../results'

# Set sigma values
sigma_values = load('{}/{}'.format(results_path, 'sigmas.npy'))

# Call the SExtractor runner
CallSepRunner(path, sigma_values)
