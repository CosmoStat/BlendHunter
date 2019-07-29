import numpy as np
import sys

bh_path = ('/home/alice/Cosmostat/Codes/BlendHunter')
sys.path.extend([bh_path])

from blendhunter.data import CreateTrainData


def get_images(sample):

    return np.array([sample[obj]['galsim_image'][0].array for obj in
                     range(sample.size)])

# def get_pos(sample):
#
#     sample[obj]['galsim_image'][0]


path = '/home/alice/Cosmostat/Codes/BlendHunter'
input = path + '/axel_sims'
output = path + '/bh'


#Getting the images
blended = np.load(input + '/blended/gal_obj_0.npy', allow_pickle=True)
not_blended = np.load(input + '/not_blended/gal_obj_0.npy', allow_pickle=True)

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

images = [get_images(sample) for sample in (blended, not_blended)]

CreateTrainData(images, output).prep_axel(fwhm, param_x, param_y, sm)
