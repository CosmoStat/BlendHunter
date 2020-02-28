#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""RUN SEXTRACTOR

This module contains methods for running SExtractor on testing data.

:Author: Samuel Farrens <samuel.farrens@cea.fr>

:Credits: Alice Lacan

"""

import numpy as np
from utils import load, DataHandler
from sep_runner import Run_Sep
from os.path import expanduser


class CallSepRunner:
    """Call SEP Runner

    This class calls the SEP (SExtractor) runner on the specified images and
    saves the output predictions.

    Parameters
    ----------
    path : str
        Path to results

    """

    def __init__(self, in_path, out_path, sigma_values, n_noise_reals=5,
                 data_dir='bh_data', sep_data_dir='sepData',
                 preds_path='sim_results', output_str='sep_preds'):

        self.in_path = in_path
        self.out_path = out_path
        self.sigma_values = sigma_values.astype(int)
        self.noise_reals = list(range(1, n_noise_reals + 1))
        self.data_dir = data_dir
        self.preds_path = preds_path
        self.sep_data_dir = sep_data_dir
        self.output_str = output_str
        self._call_sep_run()

    def _get_input_dir(self, sigma, noise_real):

        input_dir = '{}/{}_{}_{}/{}'.format(self.in_path, self.data_dir,
                                            sigma, noise_real,
                                            self.sep_data_dir)

        return input_dir

    def _load_data(self, sigma, noise_real):
        """Load Data

        This method loads the datasets in the provided input path.

        Parameters
        ----------

        """

        input_dir = self._get_input_dir(sigma, noise_real)

        return DataHandler(input_dir, sort=False).datasets

    def _run_sep(self, blended, not_blended):
        """Run SEP

        This method calls the SEP runner on a given set of images.

        Parameters
        ----------
        blended : numpy.ndarray
            Blended images
        not_blended : numpy.ndarray
            Non blended images

        Returns
        -------
        numpy.ndarray
            SEP predictions

        """

        runner = Run_Sep()

        preds_b, _ = runner.process(blended)
        preds_nb, _ = runner.process(not_blended)

        return np.concatenate((preds_b, preds_nb), axis=0)

    def _save_preds(self, preds, sigma, noise_real):
        """Save Predictions

        Save the predictions to a numpy binary file.

        Parameters
        ----------
        preds : numpy.ndarray
            Predictions

        """

        path = '{}/{}/sep_results'.format(self.out_path, self.preds_path)
        np.save('{}/{}_{}_{}'.format(path, self.output_str, sigma, noise_real),
                preds)

    def _call_sep_run(self):
        """Call SEP Runner

        This method calls the SEP runner on all the input files.

        """

        for sigma in self.sigma_values:
            for noise_real in self.noise_reals:

                preds = self._run_sep(*self._load_data(sigma, noise_real))
                self._save_preds(preds, sigma, noise_real)


# to be removed
user_home = expanduser("~")

# Set paths
results_path = '../results'
input_path_sim = user_home + '/Desktop/bh_data'
input_path_cosmos = user_home + '/Desktop/cosmos_data'

# Set sigma values
sigma_values = load('{}/{}'.format(results_path, 'sigmas.npy'))

# Call the SExtractor runner on the simulated data
CallSepRunner(input_path_sim, results_path, sigma_values)

# Call the SExtractor runner on the COSMOS data
CallSepRunner(input_path_cosmos, results_path, sigma_values, n_noise_reals=1,
              preds_path='cosmos_results')
