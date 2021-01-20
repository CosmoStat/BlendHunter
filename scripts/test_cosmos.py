import numpy as np
from blendhunter.config import BHConfig
from blendhunter.data import CreateTrainData
import script_functions as sf


def prep_cosmos(out_path, blended, not_blended, dir_str=''):

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


# Read BH configuration file
bhconfig = BHConfig().config
out_path = bhconfig['out_path']
input = bhconfig['cosmos_path']

# Getting the images
blended = np.load(input + '/blended/gal_obj_0.npy', allow_pickle=True)
not_blended = np.load(input + '/not_blended/gal_obj_0.npy', allow_pickle=True)

# prep_data(out_path, blended, not_blended)
prep_cosmos(out_path, blended, not_blended, dir_str='_pad')
