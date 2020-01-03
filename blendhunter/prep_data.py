import numpy as np
import sys
import modopt
from modopt.signal.noise import add_noise
from modopt.base.np_adjust import pad2d

from os.path import expanduser
user_home = expanduser("~")

#Check the folder hierarchy
bh_path = (user_home+'/Documents/Cosmostat/Codes/BlendHunter')
sys.path.extend([bh_path])

from blendhunter.data import CreateTrainData

#Function to add noise according to signal
def noise_to_add(img=None, SNR_=None, map=None):
    #Calculate std of central object signal
    signal = np.sum(img[map == 1]**2)
    # Calculate noise to add on image
    std_noise = np.sqrt(signal) / SNR_
    #add noise
    return add_noise(img, sigma = std_noise)

#Function to get images
def get_images(sample, add_noise_img = False, sigma_noise = None, add_padding_noise=False, fixed_snr = False, SNR_target=None, seg_map=None):

    if add_noise_img:
        #Add noise to image and store array in dict
        for i in range(len(sample)):
            sample[i]['galsim_image_noisy'] = add_noise(sample[i]['galsim_image'][0].array, sigma = sigma_noise)

        return np.array([sample[obj]['galsim_image_noisy'] for obj in
                         range(sample.size)])

    if add_padding_noise:
        #Pad image 7 by 7
        for i in range(len(sample)):
            sample[i]['galsim_image_pad'] = pad2d(sample[i]['galsim_image'][0].array, (7, 7))

        #Then add noise to image and store array in dict
        for i in range(len(sample)):
            sample[i]['galsim_image_noisy'] = add_noise(sample[i]['galsim_image_pad'], sigma = sigma_noise)

        return np.array([sample[obj]['galsim_image_noisy'] for obj in
                         range(sample.size)])
    # Add noise according to a fixed SNR level to reach
    if fixed_snr:
        for i in range(len(sample)):
            sample[i]['galsim_image_noisy'] = noise_to_add(img=sample[i]['galsim_image'][0].array, SNR_=SNR_target, map=seg_map[i])
            #Normalise images
            sample[i]['galsim_image_noisy'] /= sample[i]['galsim_image_noisy'].max()

        return np.array([sample[obj]['galsim_image_noisy'] for obj in
                         range(sample.size)])
    #Return only non noisy image
    else:
        return np.array([sample[obj]['galsim_image'][0].array for obj in
                     range(sample.size)])


#Paths for datasets with different sigma_noise
path = user_home+'/Documents/Cosmostat/Codes/BlendHunter'

input = path + '/axel_sims/larger_dataset'
input_real = path + '/axel_sims/deblending_real'
output= path + '/bh_pad14'

#Getting the images
blended = np.load(input+ '/blended/gal_obj_0.npy', allow_pickle=True)
not_blended = np.load(input+ '/not_blended/gal_obj_0.npy', allow_pickle=True)

#Get images
images = [get_images(sample, add_padding_noise = True, sigma_noise=14.0) for sample in (blended, not_blended)]
#Save noisy images for comparison w/ SExtractor
np.save(output+'/blended_noisy.npy', blended)
np.save(output+'/not_blended_noisy.npy', not_blended)

#Train-valid-test split
CreateTrainData(images, output).prep_axel(path_to_output=output)
