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
def get_images(sample, add_noise_img = False, sigma_noise = None, extra_padding=False, add_pad_noise=False, fixed_snr = False, SNR_target=None, seg_map=None):

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

    else:
        return np.array([sample[obj]['galsim_image'][0].array for obj in
                     range(sample.size)])


#Paths for datasets with different sigma_noise
path = '/Users/lacan/Documents/Cosmostat/Codes/BlendHunter'
input = path + '/axel_sims/larger_dataset'
input_real = path + '/axel_sims/deblending_real'
output= path + '/bh_{}'

#Getting the images
blended = np.load(input+ '/blended/gal_obj_0.npy', allow_pickle=True)
not_blended = np.load(input+ '/not_blended/gal_obj_0.npy', allow_pickle=True)

for j in [5, 14, 18, 26, 35, 40]:
    for i in [1,2,3,4]:
        #Get images
        images = [get_images(sample, add_pad_noise = True, sigma_noise=j) for sample in (blended, not_blended)]
        #Save noisy images for comparison w/ SExtractor
        np.save(output.format(str(j)+str(i))+'/blended_noisy{}.npy'.format(str(j)+str(i)), blended[36000:40000])
        np.save(output.format(str(j)+str(i))+'/not_blended_noisy{}.npy'.format(str(j)+str(i)), not_blended[36000:40000])
        #Train-valid-test split
        CreateTrainData(images, output.format(str(j)+str(i))).prep_axel(path_to_output=output.format(str(j)+str(i)))
