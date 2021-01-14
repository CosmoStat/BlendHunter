
import os
import numpy as np


def get_acc(dir_name, out_path, noise_sigma, n_noise_real, labels, ext='.npy',
            return_type='accuracy', average=True):

    path = os.path.join(out_path, dir_name)
    load = lambda x: np.load(x, allow_pickle=True)
    file_list = os.listdir(path)

    prefix = ''.join([char for char in file_list[0]
                      if not char.isdigit()]).rstrip(ext)

    if 'sep' in dir_name:
        labels = np.array(labels == 'blended').astype(int)

    res_sigma = []

    for sigma in noise_sigma:

        res_real = []

        for noise_real in range(n_noise_real):

            id = f'{str(sigma)}{str(noise_real)}'
            file_name = f'{prefix}{id}{ext}'
            preds = load(os.path.join(path, file_name))
            retrieval = np.array(preds == labels)
            acc = (np.sum(retrieval).astype(int).flatten() / retrieval.size)

            if return_type == 'accuracy':
                res = acc
            elif return_type == 'retrieval':
                res = retrieval

            res_real.append(res)

        res_sigma.append(res_real)

    if average:
        return [metric(res_sigma, axis=1) for metric in (np.mean, np.std)]

    else:
        return np.array(res_sigma)
