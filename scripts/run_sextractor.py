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

    def __init__(self, path, sigma_values, n_noise_reals=5, prefix='bh_pad',
                 output_str='sep_preds'):

        self.path = path
        self.sigma_values = sigma_values
        self.noise_reals = [''] + list(range(1, n_noise_reals))
        self.prefix = prefix
        self.output_str = output_str
        self._call_sep_run()

    def _load_data(self, sigma, noise_real):
        """Load Data

        This method loads the datasets in the provided input path.

        Parameters
        ----------

        """

        return DataHandler('{}/{}{}{}'.format(self.path, self.prefix,
                           sigma, noise_real)).datasets

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

        # to be removed
        path = '../results/sep_results'
        np.save('{}/{}_{}_{}'.format(path, self.output_str, sigma, noise_real),
                preds)

        # np.save('{}/{}_{}_{}'.format(self.path, self.output_str, sigma,
        #         noise_real),
        #         preds)

    def _call_sep_run(self):
        """Call SEP Runner

        This method calls the SEP runner on all the input files.

        """

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
