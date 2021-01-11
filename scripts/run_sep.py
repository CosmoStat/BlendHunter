"""Run SExtractor

Run SExtracton script (written by A. Guinot, github.com/aguinot) on padded and
un-padded noisy images.

"""

import numpy as np
from sep_script import Run_Sep
from blendhunter.config import BHConfig


def sep_results(out_path, blends=None, no_blends=None, sigma_val=None,
                dir_str='bh_', verbose=True):
    """SEP Results

    Notes
    -----
    blends=blended images
    no_blends=non blended images
    """
    suffix = '_pad' if 'pad' in dir_str else ''

    sep_runner = Run_Sep()
    flags_b, sep_res_b = sep_runner.process(blends)
    flags_nb, sep_res_nb = sep_runner.process(no_blends)

    # Compute accuracy
    acc = ((len(np.where(flags_b == 1)[0]) +
            len(np.where(flags_nb == 0)[0])) /
           (len(flags_b) + len(flags_nb)))

    # Concatenate flags
    flags = np.concatenate((flags_b, flags_nb), axis=0)

    np.save(out_path + f'/sep_results{suffix}/flags{sigma_val}.npy', flags)

    if verbose:
        print(f'Sep Accuracy (sigma_noise = {sigma_val}): {acc * 100}%')

    n_miss = ((len(np.where(flags_b == 16)[0]) +
               len(np.where(flags_nb == 16)[0])) /
              (len(flags_b)+len(flags_nb)))

    if verbose:
        print(f'Misidentified : {n_miss * 100}%')

    return flags


def get_sep_results(out_path, noise_sigma, n_noise_real, dir_str='bh_',
                    verbose=True):

    for sigma in noise_sigma:
        for noise_real in range(n_noise_real):

            id = f'{str(sigma)}{str(noise_real)}'
            path = f'{out_path}/{dir_str}{id}'

            if verbose:
                print(f'Processing {dir_str}{id}')

            blends = np.load(f'{path}/blended_noisy{id}.npy',
                             allow_pickle=True)
            no_blends = np.load(f'{path}/not_blended_noisy{id}.npy',
                                allow_pickle=True)
            res = sep_results(out_path, blends, no_blends, sigma, dir_str,
                              verbose)


# Read BH configuration file
bhconfig = BHConfig().config
out_path = bhconfig['out_path']
noise_sigma = bhconfig['noise_sigma']
n_noise_real = bhconfig['n_noise_real']

# Run SExtractor on non padded images
# get_sep_results(out_path, noise_sigma, n_noise_real)

# Run SExtractor on padded images
get_sep_results(out_path, noise_sigma, n_noise_real, dir_str='bh_pad')
