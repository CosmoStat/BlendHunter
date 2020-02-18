#! /usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np


def load(path):
    """Load

    Load numpy binary with allow_pickle set to True.

    Parameters
    ----------
    path : str
        Path to file

    Returns
    -------
    numpy.ndarray
        Array loaded from numpy binary

    """

    return np.load(path, allow_pickle=True)


class GetAcc:
    """Get Accuracy

    Class to calculate the average classification accuracy for a range of noise
    values and a series of noise realisations.

    Parameters
    ----------
    path : str
        Input path
    sigma_values : numpy.ndarray
        Noise standard deviation values
    labels : numpy.ndarray
        True classification labels
    prefix : str
        Input file prefix string
    n_noise_reals : int
        Number of noise realisations

    """

    def __init__(self, path, sigma_values, labels, prefix='preds',
                 n_noise_reals=5):

        self.path = path
        self.sigma_values = sigma_values
        self.labels = labels
        self.prefix = prefix
        self.noise_reals = np.array([''] + list(range(1, n_noise_reals)))
        self._get_stats()

    def _load_dataset(self, sigma, real):
        """Load Dataset

        Parameters
        ----------
        sigma : float
            Noise standard deviation
        real : int
            Realisation index

        Returns
        -------
        numpy.ndarray
            Array loaded from numpy binary

        """

        return load('{}/{}{}{}.npy'.format(self.path, self.prefix, int(sigma),
                    real))

    def _get_acc(self, dataset):
        """Get Accuracy

        Calculate the classification accuracy for a given dataset.

        Parameters
        ----------
        dataset : numpy.ndarray
            Set of predictions for a given dataset

        Returns
        -------
        float
            Classification accuracy

        """

        if self.prefix == 'preds':

            return np.sum(dataset == self.labels) / self.labels.size

        else:

            return ((len(np.where(dataset[0:4000] == 1)[0]) +
                    len(np.where(dataset[4000:8000] == 0)[0])) /
                    (len(dataset[0:4000])+len(dataset[4000:8000])))

    def _get_stats(self):
        """Get Statistics

        Calculate the mean and standard deviation of the classification
        accuracy for a given set of noise realisations.

        """

        res = np.array([self._get_acc(self._load_dataset(sigma, real))
                        for sigma in self.sigma_values
                        for real in self.noise_reals]).reshape(
                        self.sigma_values.size, self.noise_reals.size)

        self.mean = np.mean(res, axis=1)
        self.std = np.std(res, axis=1)


# Set output path
results_path = '../results'

# Set sigma values and load true classification labels
sigma_values = np.array([5.0, 14.0, 18.0, 26.0, 35.0, 40.0])
labels = load('{}/{}'.format(results_path, 'labels.npy'))

# Get classification accuracy results for BlendHunter
bh_res = GetAcc(results_path + '/bh_results', sigma_values, labels)

# Get classification accuracy results for SExtractor
se_res = GetAcc(results_path + '/se_results', sigma_values, labels,
                prefix='flags_pad')

# Save noise standard deviation values
np.save(results_path + '/sigmas', sigma_values)

# Save classification accuracy results
np.save(results_path + '/acc_results', np.array([bh_res.mean, se_res.mean,
                                                bh_res.std, se_res.std]))
