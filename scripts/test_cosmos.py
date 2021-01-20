import numpy as np
from blendhunter.config import BHConfig
from blendhunter.data import CreateTrainData
from blendhunter import BlendHunter
import script_functions as sf


def prep_cosmos(out_path, blended, not_blended, dir_str='', verbose=True):

    path = f'{out_path}/cosmos_data{dir_str}'
    pad = 'pad' in dir_str

    if verbose:
        print(f'Processing {path}')

    images = [sf.get_images(sample, add_padding=pad, add_noise_sigma=False)
              for sample in (blended, not_blended)]

    # Save noisy test images for comparison w/ SExtractor
    np.save(f'{path}/blended.npy', blended)
    np.save(f'{path}/not_blended.npy', not_blended)

    # BH Test data
    CreateTrainData(images, path, train_fractions=(0, 0, 1)).prep_cosmos()


def get_bh_preds(out_path, noise_sigma, n_noise_real, dir_str='bh_',
                 verbose=True):

    dirpad = '_pad' if 'pad' in dir_str else ''

    for sigma in noise_sigma:
        for noise_real in range(n_noise_real):

            id = f'{str(sigma)}{str(noise_real)}'
            path = f'{out_path}/{dir_str}{id}'

            if verbose:
                print(f'Using {dir_str}{id}')

            bh = BlendHunter(weights_path=path + '/weights')

            # Predict Results
            pred_top = bh.predict(out_path + f'/cosmos_data{dirpad}' +
                                  '/BlendHunterData/test/test',
                                  weights_type='top')

            if verbose:
                true = np.load(path + '/BlendHunterData/test/test/labels.npy')
                print("Match Top:", np.sum(pred_top == true) / true.size)
                print("Error Top", np.sum(pred_top != true) / true.size)

            # Save history and results
            np.save(out_path + f'/cosmos_results{dirpad}/preds{id}.npy',
                    pred_top)


# Read BH configuration file
bhconfig = BHConfig().config
out_path = bhconfig['out_path']
input = bhconfig['cosmos_path']
noise_sigma = bhconfig['noise_sigma']
n_noise_real = bhconfig['n_noise_real']
cosmos_sample_range = slice(*bhconfig['cosmos_sample_range'])

# Getting the images
blended = np.load(input + '/blended/gal_obj_0.npy',
                  allow_pickle=True)[cosmos_sample_range]
not_blended = np.load(input + '/not_blended/gal_obj_0.npy',
                      allow_pickle=True)[cosmos_sample_range]

# prep_data(out_path, blended, not_blended)
# prep_cosmos(out_path, blended, not_blended, dir_str='_pad')

# get_bh_preds(out_path, noise_sigma, n_noise_real)
get_bh_preds(out_path, noise_sigma, n_noise_real, dir_str='bh_pad')
