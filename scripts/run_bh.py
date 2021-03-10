import numpy as np
from blendhunter.config import BHConfig
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

# Set plaidml backend for Keras before importing blendhunter
import plaidml.keras
plaidml.keras.install_backend()

# Import BlendHunter
from blendhunter import BlendHunter


def run_bh(out_path, noise_sigma, n_noise_real, dir_str='bh_',
           verbose=True):

    dirpad = '_pad' if 'pad' in dir_str else ''

    for sigma in noise_sigma:
        for noise_real in range(n_noise_real):

            id = f'{str(sigma)}{str(noise_real)}'
            path = f'{out_path}/{dir_str}{id}'

            if verbose:
                print(f'Processing {dir_str}{id}')

            bh = BlendHunter(weights_path=path + '/weights')

            # Train Network (no fine tuning)
            bh.train(
             path + '/BlendHunterData',
             get_features=True,
             train_top=True,
             fine_tune=False
            )

            # Get training history
            hist = np.array(bh.history.history)

            # Predict Results
            pred_top = bh.predict(path + '/BlendHunterData/test/test',
                                  weights_type='top')

            if verbose:
                true = np.load(path + '/BlendHunterData/test/test/labels.npy')
                print("Match Top:", np.sum(pred_top == true) / true.size)
                print("Error Top", np.sum(pred_top != true) / true.size)

            # Save history and results
            np.save(path + '/BlendHunterData/test/test/history.npy', hist)
            np.save(out_path + f'/bh{dirpad}_results/preds{id}.npy', pred_top)


# Read BH configuration file
bhconfig = BHConfig().config
out_path = bhconfig['out_path']
noise_sigma = bhconfig['noise_sigma']
n_noise_real = bhconfig['n_noise_real']

# Prepare non padded images
# run_bh(out_path, noise_sigma, n_noise_real)

# Prepare padded images
run_bh(out_path, noise_sigma, n_noise_real, dir_str='bh_pad')
