import numpy as np
import sys
import modopt
from modopt.signal.noise import add_noise
from modopt.base.np_adjust import pad2d

from os.path import expanduser
user_home = expanduser("~")
#Check for the folder hierarchy
bh_path = (user_home+'/Cosmostat/Codes/BlendHunter')
sys.path.extend([bh_path])

#Import the class to split into train-valid-test
from blendhunter.data import CreateTrainData

#Function to get images
def get_images(sample, add_noise_img = False, sigma_noise = None, add_padding_noise=False, fixed_snr = False, SNR_target=None, seg_map=None):
    """1. To pad with noise, use 'add_padding_noise=True' + 'sigma_noise'
       2. To simply add noise, use 'add_noise_img=True' + 'sigma_noise'
       3. Otherwise, the function retrieves non noisy simulations by default"""

    if add_noise_img:
        #Add noise to image and store array in dict
        for i in range(len(sample)):
            sample[i]['galsim_image_noisy'] = add_noise(sample[i]['galsim_image'][0].array, sigma = sigma_noise)

        return np.array([sample[obj]['galsim_image_noisy'] for obj in
                         range(sample.size)])
    #Padding of 7 by 7 + noise
    if add_padding_noise:
        #Pad image
        for i in range(len(sample)):
            sample[i]['galsim_image_pad'] = pad2d(sample[i]['galsim_image'][0].array, (7, 7))

        #Then add noise to image and store array in dict
        for i in range(len(sample)):
            sample[i]['galsim_image_noisy'] = add_noise(sample[i]['galsim_image_pad'], sigma = sigma_noise)

        return np.array([sample[obj]['galsim_image_noisy'] for obj in
                         range(sample.size)])

    #Get non noisy images
    else:
        return np.array([sample[obj]['galsim_image'][0].array for obj in
                     range(sample.size)])


#Paths for datasets with different sigma_noise
path = user_home+'/Cosmostat/Codes/BlendHunter'
#Path for simulations
input = path + '/axel_sims/larger_dataset'

"""Datasets are called bh_+ the noise level + the number of noise realisation"""
output_pad = path + '/bh_pad{}'
output = path + '/bh_{}'

#Getting the simulations
blended = np.load(input+ '/blended/gal_obj_0.npy', allow_pickle=True)
not_blended = np.load(input+ '/not_blended/gal_obj_0.npy', allow_pickle=True)

"""Loop to split padded and noisy datasets in train-valid-test"""
for j in [3,7,10,12,16,20,22,24,28,30,32,37,42,44]:
    #Get images
    images = [get_images(sample, add_padding_noise = True, sigma_noise=j) for sample in (blended, not_blended)]
    #Save noisy test images for comparison w/ SExtractor
    #np.save(output_pad.format(str(j)+str(i))+'/blended_noisy{}.npy'.format(str(j)+str(i)), blended[36000:40000])
    #np.save(output_pad.format(str(j)+str(i))+'/not_blended_noisy{}.npy'.format(str(j)+str(i)), not_blended[36000:40000])
    #Train-valid-test split
    CreateTrainData(images, output_pad.format(str(j)+str(i))).prep_axel(path_to_output=output_pad.format(str(j)+str(i)))
