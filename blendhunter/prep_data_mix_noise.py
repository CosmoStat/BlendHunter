import numpy as np
import sys
import modopt
from modopt.signal.noise import add_noise
from modopt.base.np_adjust import pad2d

bh_path = ('/Users/lacan/Documents/Cosmostat/Codes/BlendHunter')
sys.path.extend([bh_path])

from blendhunter.data import CreateTrainData

def noise_to_add(img=None, SNR_=None, map=None):
    #Calculate std of central object signal
    signal = np.sum(img[map == 1]**2)
    # Calculate noise to add on image
    std_noise = np.sqrt(signal) / SNR_
    #add noise
    return add_noise(img, sigma = std_noise)

#Function to get images
def get_images(sample, add_noise_img = False, sigma_noise = None, extra_padding=False,
               add_pad_noise=False, fixed_snr = False, SNR_target=None, seg_map=None, get_noisy=False):

    if add_noise_img:
        #Add noise to image and store array in dict
        for i in range(len(sample)):
            sample[i]['galsim_image_noisy'] = add_noise(sample[i]['galsim_image'][0].array, sigma = sigma_noise)

        return np.array([sample[obj]['galsim_image_noisy'] for obj in
                         range(sample.size)])

    if extra_padding:
        #Add extra padding to image and store in dict
        for i in range(len(sample)):
            sample[i]['galsim_image_pad'] = pad2d(sample[i]['galsim_image'][0].array, (7, 7))

        return np.array([sample[obj]['galsim_image_pad'] for obj in
                         range(sample.size)])

    if add_pad_noise:
        #Pad image
        for i in range(len(sample)):
            sample[i]['galsim_image_pad'] = pad2d(sample[i]['galsim_image'][0].array, (7, 7))

        #Then add noise to image and store array in dict
        for i in range(len(sample)):
            sample[i]['galsim_image_noisy'] = add_noise(sample[i]['galsim_image_pad'], sigma = sigma_noise)

        return np.array([sample[obj]['galsim_image_noisy'] for obj in
                         range(sample.size)])

    if fixed_snr:
        for i in range(len(sample)):
            sample[i]['galsim_image_noisy'] = noise_to_add(img=sample[i]['galsim_image'][0].array, SNR_=SNR_target, map=seg_map[i])
            #Normalise images
            sample[i]['galsim_image_noisy'] /= sample[i]['galsim_image_noisy'].max()

        return np.array([sample[obj]['galsim_image_noisy'] for obj in
                         range(sample.size)])

    if get_noisy:
        return np.array([sample[obj]['galsim_image_noisy'] for obj in
                         range(sample.size)])

    else:
        return np.array([sample[obj]['galsim_image'][0].array for obj in
                     range(sample.size)])

#Create dataset with same iages and different level of noise
#Import images and select which one to repeat

def get_sub_sample(nb_img_in_sample=10000, path=None):
    #Import data
    sub_sample = np.load(path, allow_pickle=True)
    sub_sample = sub_sample[0:nb_img_in_sample]

    return sub_sample

def create_mix_noise_dataset(paths):
     #Get the sub samples
     samples = [get_sub_sample(path=i) for i in paths]
     #concatenate the arrays
     final_sample = np.array([samples[j][i] for i in range(len(samples[0])) for j in range(len(samples))])

     return final_sample


#Paths for datasets with different sigma_noise
path = '/Users/lacan/Documents/Cosmostat/Codes/BlendHunter'
input = path + '/axel_sims/larger_dataset'
input_real = path + '/axel_sims/deblending_real'
output = path + '/bh_mix_noise'

path_to_inputb = path +'/blended_image_noisy/blended_noisy{}.npy'
path_to_inputnb = path +'/not_blended_image_noisy/not_blended_noisy{}.npy'

paths_to_input = ([path_to_inputb.format(i) for i in [5,14,26,35]],[path_to_inputnb.format(i) for i in [5,14,26,35]])

#Getting the images
blended = create_mix_noise_dataset(paths = paths_to_input[0])
not_blended = create_mix_noise_dataset(paths = paths_to_input[1])

#Importing segmentation maps (blended and not blended)
#sm_b = np.array([blended[val]['seg_map'][0].array for val in range(blended.size)])
#sm_nb = np.array([not_blended[val]['seg_map'][0].array for val in range(not_blended.size)])

#sm=np.concatenate((sm_b, sm_nb), axis = 0)

#Getting the psf (fwhm)
#fwhm_blended = np.array([blended[val]['PSF']['fwhm'] for val in range(blended.size)])
#fwhm_not_blended = np.array([not_blended[val]['PSF']['fwhm'] for val in range(not_blended.size)])

#Final fwhm array
#fwhm = [fwhm_blended, fwhm_not_blended]

#Getting the shift parameters
#param_x = [np.array([blended[val]['blend_param']['dx'] for val in range(blended.size)])]
#param_y = [np.array([blended[val]['blend_param']['dy'] for val in range(blended.size)])]

#Getting the segmentation map
#sm = [np.array([pad2d(blended[val]['seg_map'][0].array, (7,7)) for val in range(blended.size)])]

#Get images
images = [get_images(sample, get_noisy=True ) for sample in (blended, not_blended)]

#Save noisy images for comparison w/ SExtractor
np.save(output+'/blended_noisy.npy', blended)
#np.save(output_real+'/not_blended_noisy.npy', not_blended)

#Train-valid-test split
CreateTrainData(images, output).prep_axel(path_to_output=output)
