import numpy as np
import sys
from os.path import expanduser
user_home = expanduser("~")

bh_path = (user_home+'/Cosmostat/Codes/BlendHunter')
sys.path.extend([bh_path])

from blendhunter import BlendHunter

"""Predict using given pretrained weights.The function returns accuracy.()"""
def run_bh(path, value):

    w_path = user_home+'/Cosmostat/Codes/BlendHunter/pretrained_weights_pad'

    bh = BlendHunter(weights_path=w_path+'/weights{}'.format(value))
    """Only testing on the test images, no training"""
    bh.train(path + '/BlendHunterData',
         get_features=False,
         train_top=False,
         fine_tune=False)

    """Predict Results"""
    pred_top = bh.predict(path + '/BlendHunterData/test/test', weights_type='top')
    true = np.load(path + '/BlendHunterData/test/test/labels.npy')
    print("Match Top:", np.sum(pred_top == true) / true.size)
    print("Error Top", np.sum(pred_top != true) / true.size)
    np.save(path+'/preds_real_data{}.npy'.format(value), pred_top)


    return np.sum(pred_top == true) / true.size


"""Check the folder hierarchy"""
path = user_home+'/Cosmostat/Codes/BlendHunter'
#Path for simulations
input = path + '/axel_sims/real_deblending'
output = path+'/bh_real_pad'
#output = path+'/bh_real'

"""Getting the simulations"""
"""Check that folders names in /axel_sims"""
blended = np.load(input+ '/blended/gal_obj_0.npy', allow_pickle=True)
not_blended = np.load(input+ '/not_blended/gal_obj_0.npy', allow_pickle=True)

from prep_data_loop import get_images

"""Generate the dataset for padded images"""
images = [get_images(sample, add_padding_noise = True, sigma_noise=j) for sample in (blended, not_blended)]
#Save noisy test images for comparison w/ SExtractor
np.save(output.format(str(j)+str(i))+'/blended_noisy.npy', blended[36000:40000])
np.save(output.format(str(j)+str(i))+'/not_blended_noisy.npy', not_blended[36000:40000])
#Train-valid-test split
CreateTrainData(images, output.format(str(j)+str(i))).prep_axel(path_to_output=output.format(str(j)+str(i)))

#"""Generate the dataset for non padded images"""
#images = [get_images(sample, add_noise_img = True, sigma_noise=j) for sample in (blended, not_blended)]
#Save noisy test images for comparison w/ SExtractor
#np.save(output.format(str(j)+str(i))+'/blended_noisy{}.npy'.format(str(j)+str(i)), blended[36000:40000])
#np.save(output.format(str(j)+str(i))+'/not_blended_noisy{}.npy'.format(str(j)+str(i)), not_blended[36000:40000])
#Train-valid-test split
#CreateTrainData(images, output.format(str(j)+str(i))).prep_axel(path_to_output=output.format(str(j)+str(i)))

"""Run the network on real data"""

sigmas = [5,14,18,26,35,40]
noise_realisation = ['',1,2,3,4]
datasets = [str(j)+str(i) for j in sigmas for i in noise_realisation]

for i in datasets:
    """Compute accuracy for each set of weights"""
    acc = run_bh(output, i)
    print('Accuracy of {}% for the weights{}'.format(acc, str(i)))

"""Run sep"""
"""Import script for SExtractor
Script written by A. Guinot : https://github.com/aguinot"""
from sep_script import Run_Sep

#SEP params
BIG_DISTANCE = 1e30
NO_BLEND = 0
BLEND = 1
MISS_EXTRACTION = 16

def sep_results(blends=None,no_blends=None,path=None):
    """blends= blended images
       no_blends= non blended images"""
    runner = Run_Sep()
    flags_b, sep_res_b = runner.process(blends) #We retrieve sep predictions and positions of blends
    flags_nb, sep_res_nb = runner.process(no_blends)

    #Compute accuracy
    acc = (len(np.where(flags_b == 1)[0])+len(np.where(flags_nb == 0)[0]))/(len(flags_b)+len(flags_nb))

    #Concatenate flags
    flags = np.concatenate((flags_b, flags_nb), axis =0)
    #sep_res = np.concatenate((sep_res_b, sep_res_nb), axis =0)

    """Make sure the 'sep_pad_results' folder exists"""
    #save
    np.save(path+'/sep_pad_results/flags_real_pad.npy', flags)
    #np.save(path+'/sep_results_pad/sep_res{}.npy'., sep_res)
    print('Sep Accuracy: {}%'., acc*100))
    n_miss = (len(np.where(flags_b == 16)[0])+len(np.where(flags_nb == 16)[0]))/(len(flags_b)+len(flags_nb))
    print('Misidentified : {}%'.format(n_miss*100))

    return flags

#Getting the test images
blended = np.load(output+'/blended_noisy.npy', allow_pickle=True)
not_blended = np.load(output+'/not_blended_noisy.npy', allow_pickle=True)
####Run sep
sep_results(blends= blended[i][j], no_blends= not_blended[i][j] , path = path)
