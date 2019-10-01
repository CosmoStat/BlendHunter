import numpy as np
import sys
import modopt
from modopt.signal.noise import add_noise

bh_path = ('/Users/alacan/Documents/Cosmostat/Codes/BlendHunter')
sys.path.extend([bh_path])

from blendhunter.data import CreateTrainData

#Function to get images
def get_images(sample, add_noise = False, sigma_noise = None):

    if add_noise:
        #Add noise to image and store array in dict
        for i in range(len(sample)):
            sample[i]['galsim_image_noisy'] = add_noise(sample[i]['galsim_image'][0].array, sigma = sigma_noise)

        return np.array([sample[obj]['galsim_image_noisy'] for obj in
                         range(sample.size)])
    else:
        return np.array([sample[obj]['galsim_image'][0].array for obj in
                     range(sample.size)])


#Paths for datasets with different sigma_noise
path = '/Users/alacan/Documents/Cosmostat/Codes/BlendHunter'
input = path + '/axel_sims/larger_dataset'
input_real = path + '/axel_sims/deblending_real'
output_real = path + '/bh_real'
output = path + '/bh'
output44 = path + '/more_noise_ranges/bh_44'

#Getting the images
blended = np.load(input_real + '/blend/gal_obj_0.npy', allow_pickle=True)
not_blended = np.load(input_real + '/no_blend/gal_obj_0.npy', allow_pickle=True)


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

#Get images
images = [get_images(sample) for sample in (blended, not_blended)]

#Save noisy images for comparison w/ SExtractor
#np.save(output_real+'/blended_noisy.npy', blended)
#np.save(output_real+'/not_blended_noisy.npy', not_blended)

#Train-valid-test split
CreateTrainData(images, output_real).prep_axel(path_to_output=output_real, map=sm, param_1=param_x, param_2=param_y, psf=fwhm)
