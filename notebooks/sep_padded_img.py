###### Running SExtractor on simulated galsim images

## Import script for SExtractor
#Script written by A. Guinot : https://github.com/aguinot

import numpy as np
from sep_script import Run_Sep

#SEP params
BIG_DISTANCE = 1e30
NO_BLEND = 0
BLEND = 1
MISS_EXTRACTION = 16

from os.path import expanduser
user_home = expanduser("~")
testpath = user_home+'/Documents/Cosmostat/Codes/BlendHunter'

def extract_test_images(path):
    img = np.load(path, allow_pickle=True)
    test_img = img[36000:40000]
    return test_img


#Import function
def import_(path):
    img = np.load(path, allow_pickle=True)
    return img


# Run sep function
def sep_results(blends=None,no_blends=None, sigma_val=None,path=None):
    runner = Run_Sep()
    flags_b, sep_res_b = runner.process(blends)
    flags_nb, sep_res_nb = runner.process(no_blends)
    
    #Display results
    acc = (len(np.where(flags_b == 1)[0])+len(np.where(flags_nb == 0)[0]))/(len(flags_b)+len(flags_nb))
    
    #concatenate flags
    flags = np.concatenate((flags_b, flags_nb), axis =0)
    #sep_res = np.concatenate((sep_res_b, sep_res_nb), axis =0)
    
    #save (create 'sep_results_8000' folder)
    np.save(path+'/sep_results_pad/flags_pad{}.npy'.format(sigma_val), flags)
    #np.save(path+'/sep_results_8000/sep_res{}.npy'.format(sigma_val), sep_res)
    print('Sep Accuracy (sigma_noise = {}): {}%'.format(sigma_val, acc*100))
    n_miss = (len(np.where(flags_b == 16)[0])+len(np.where(flags_nb == 16)[0]))/(len(flags_b)+len(flags_nb))
    print('Misidentified : {}%'.format(n_miss*100))
    
    return flags

#Run sep on all noise realisations
sigmas = [5,14,18,26,35,40]
noise_realisation = ['',1,2,3,4]

datasets = [[str(j)+str(i) for i in noise_realisation]  for j in sigmas]

paths = [[[testpath+'/bh_{}/blended_noisy{}.npy'.format(i,i) for i in datasets[j]],
        [testpath+'/bh_{}/not_blended_noisy{}.npy'.format(i,i) for i in datasets[j]]] for j in range(len(datasets))]

#Getting the test images
blended = [[import_(paths[i][0][k]) for k in range(len(noise_realisation))] for i in range(len(sigmas))]
not_blended = [[import_(paths[i][1][k]) for k in range(len(noise_realisation))] for i in range(len(sigmas))]
#print('Got images for sigma_noise = {}'.format(i))
    
####Run sep
for i in range(len(sigmas)):
    for j,k in zip(range(len(noise_realisation)), datasets[i]):
        sep_results(blends= blended[i][j], no_blends= not_blended[i][j], sigma_val = k , path = testpath   )
    
    