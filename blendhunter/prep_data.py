import numpy as np
import sys
import modopt
from modopt.signal.noise import add_noise

bh_path = ('/Users/alacan/Documents/Cosmostat/Codes/BlendHunter')
sys.path.extend([bh_path])

from blendhunter.data import CreateTrainData

#Function to get images
def get_images(sample):

    return np.array([sample[obj]['galsim_image'][0].array for obj in
                     range(sample.size)])

#Second function to obtain noisy images
def get_images_noisy(sample):

    return np.array([sample[obj]['galsim_image_noisy'] for obj in
                     range(sample.size)])

#Paths for datasets with different sigma_noise
path = '/Users/alacan/Documents/Cosmostat/Codes/BlendHunter'
input = path + '/axel_sims/larger_dataset'
output = path + '/bh'
output2 = path + '/bh_5'
output3 = path + '/bh_14'
output4 = path + '/bh_18'
output5 = path + '/bh_22'
output6 = path + '/bh_26'
output7 = path + '/bh_35'
output8 = path + '/bh_40'


#Getting the images
blended = np.load(input + '/blended/gal_obj_0.npy', allow_pickle=True)
not_blended = np.load(input + '/not_blended/gal_obj_0.npy', allow_pickle=True)

#Add noise to images and store it in dict
for i in range(len(blended)):
    blended[i]['galsim_image_noisy'] = add_noise(blended[i]['galsim_image'][0].array, sigma = 35.0)
for i in range(len(not_blended)):
    not_blended[i]['galsim_image_noisy'] = add_noise(not_blended[i]['galsim_image'][0].array, sigma = 35.0)


#Getting the psf (fwhm)
fwhm_blended = np.array([blended[val]['PSF']['fwhm'] for val in range(blended.size)])
fwhm_not_blended = np.array([not_blended[val]['PSF']['fwhm'] for val in range(not_blended.size)])

#Final fwhm array
fwhm = [fwhm_blended, fwhm_not_blended]

#Getting the shift parameters
param_x = [np.array([blended[val]['blend_param']['dx'] for val in range(blended.size)])]
param_y = [np.array([blended[val]['blend_param']['dy'] for val in range(blended.size)])]

#Getting the segmentation map
sm = [np.array([blended[val]['seg_map'][0].array for val in range(blended.size)])]

#Reshape
#shift_params = [np.reshape(shift_params, (10000,1))]
#shift_params = [np.concatenate((param_x, param_y), axis=1)]

images = [get_images_noisy(sample) for sample in (blended, not_blended)]

#images_noisy = [add_noise(images[i], sigma = 22.0) for i in range(len(images))]

#Save noisy images for comparison w/ Sextractor
np.save(output7+'/blended_noisy.npy', blended)
np.save(output7+'/not_blended_noisy.npy', not_blended)


#CreateTrainData(images, output8).prep_axel(output8, fwhm, param_x, param_y, sm)
