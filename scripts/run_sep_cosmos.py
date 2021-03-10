"""Run SExtractor

Run SExtracton script (written by A. Guinot, github.com/aguinot) on padded and
un-padded noisy images.

"""

import numpy as np
from sep_script import Run_Sep
from blendhunter.config import BHConfig


def sep_results(in_path, blends, no_blends, id, dir_str='bh_', verbose=True):
    """SEP Results

    Notes
    -----
    blends=blended images
    no_blends=non blended images
    """
    dirpad = '_pad' if 'pad' in dir_str else ''

    sep_runner = Run_Sep()
    flags_b, sep_res_b = sep_runner.process(blends)
    flags_nb, sep_res_nb = sep_runner.process(no_blends)

    # Compute accuracy
    acc = ((len(np.where(flags_b == 1)[0]) +
            len(np.where(flags_nb == 0)[0])) /
           (len(flags_b) + len(flags_nb)))

    # Concatenate flags
    flags = np.concatenate((flags_b, flags_nb), axis=0)

    np.save(f'./sep{dirpad}_results/cosmos_flags.npy', flags)

    #concatenate full Results
    sep_res = np.concatenate((sep_res_b, sep_res_nb), axis=0)
    np.save(f'cosmos_sep_res.npy', sep_res)


    if verbose:
        print(f'Sep Accuracy Cosmos: {acc * 100}%')

    n_miss = ((len(np.where(flags_b == 16)[0]) +
               len(np.where(flags_nb == 16)[0])) /
              (len(flags_b)+len(flags_nb)))

    if verbose:
        print(f'Misidentified : {n_miss * 100}%')

    return flags


def get_sep_results_cosmos(in_path,dir_str='bh_', verbose=True):

    blends = np.load(f'{in_path}/blended/gal_obj_0.npy',
                     allow_pickle=True)
    no_blends = np.load(f'{in_path}/not_blended/gal_obj_0.npy',
                        allow_pickle=True)
    res = sep_results(in_path, blends, no_blends, id, dir_str,
                      verbose)

# Read BH configuration file
bhconfig = BHConfig().config
in_path = bhconfig['cosmos_path']
print(in_path)

# Run SExtractor on non padded images
# get_sep_results(out_path, noise_sigma, n_noise_real)

# Run SExtractor on padded images
get_sep_results_cosmos(in_path, dir_str='bh_pad')
