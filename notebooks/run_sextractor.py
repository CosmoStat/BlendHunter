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

#Path to images
testpath = '/Users/lacan/Documents/Cosmostat/Codes/BlendHunter'

def extract_test_images(path):
    img = np.load(path, allow_pickle=True)
    test_img = img[36000:40000]
    return test_img


# Run sep function
def sep_results(blends=None,no_blends=None, sigma_val=None,path=None):
    runner = Run_Sep()
    flags_b, sep_res_b = runner.process(blends)
    flags_nb, sep_res_nb = runner.process(no_blends)
    
    #Display results
    acc = (len(np.where(flags_b == 1)[0])+len(np.where(flags_nb == 0)[0]))/(len(flags_b)+len(flags_nb))
    
    #concatenate flags
    flags = np.concatenate((flags_b, flags_nb), axis =0)
    sep_res = np.concatenate((sep_res_b, sep_res_nb), axis =0)
    
    #save (create 'sep_results_8000' folder)
    np.save(path+'/sep_results_8000/flags{}.npy'.format(sigma_val), flags)
    np.save(path+'/sep_results_8000/sep_res{}.npy'.format(sigma_val), sep_res)
    print('Sep Accuracy (sigma{}): {}%'.format(sigma_val, acc*100))
    n_miss = (len(np.where(flags_b == 16)[0])+len(np.where(flags_nb == 16)[0]))/(len(flags_b)+len(flags_nb))
    print('Misidentified : {}%'.format(n_miss*100))
    
    return flags

sigmas = np.array([[5,51,52 ,53, 54], [14,141,142,143,144], [18,181,182,183,184],
                  [26,261,262,263,264], [35,351,352,353,354], [40,401,402,403,404]])

'''1. $\sigma_{noise} = 5$'''
################## #######################################################
paths5 = np.array([[testpath+'/images_noisy/blended_noisy{}.npy'.format(i) for i in sigmas[0]],
                   [testpath+'/images_noisy/not_blended_noisy{}.npy'.format(i) for i in sigmas[0]]])
#Getting the images 
blended_5 = [extract_test_images(paths5[0][j]) for j in range(5)]
not_blended_5 = [extract_test_images(paths5[1][j]) for j in range(5)]

####Run sep
for i,j in zip(range(5), sigmas[0]):
    sep_results(blends=blended_5[i], no_blends = not_blended_5[i], sigma_val=j, path=testpath)
    
'''2. $\sigma_{noise} = 14$'''    
###################################################################
paths14 = np.array([[testpath+'/images_noisy/blended_noisy{}.npy'.format(i) for i in sigmas[1]],
                   [testpath+'/images_noisy/not_blended_noisy{}.npy'.format(i) for i in sigmas[1]]])
#Getting the images 
blended_14 = [extract_test_images(paths14[0][j]) for j in range(5)]
not_blended_14 = [extract_test_images(paths14[1][j]) for j in range(5)]

####Run sep
for i,j in zip(range(5), sigmas[1]):
    sep_results(blends=blended_14[i], no_blends = not_blended_14[i], sigma_val=j, path=testpath)
    
'''3. $\sigma_{noise} = 18$'''    
##################################################################### 
paths18 = np.array([[testpath+'/images_noisy/blended_noisy{}.npy'.format(i) for i in sigmas[2]],
                   [testpath+'/images_noisy/not_blended_noisy{}.npy'.format(i) for i in sigmas[2]]])
#Getting the images 
blended_18 = [extract_test_images(paths18[0][j]) for j in range(5)]
not_blended_18 = [extract_test_images(paths18[1][j]) for j in range(5)]

####Run sep
for i,j in zip(range(5), sigmas[2]):
    sep_results(blends=blended_18[i], no_blends = not_blended_18[i], sigma_val=j, path=testpath)

'''4. $\sigma_{noise} = 26$'''
#################################################################### 
paths26 = np.array([[testpath+'/images_noisy/blended_noisy{}.npy'.format(i) for i in sigmas[3]],
                   [testpath+'/images_noisy/not_blended_noisy{}.npy'.format(i) for i in sigmas[3]]])
#Getting the images 
blended_26 = [extract_test_images(paths26[0][j]) for j in range(5)]
not_blended_26 = [extract_test_images(paths26[1][j]) for j in range(5)]

####Run sep
for i,j in zip(range(5), sigmas[3]):
    sep_results(blends=blended_26[i], no_blends = not_blended_26[i], sigma_val=j, path=testpath)

'''4. $\sigma_{noise} = 26$'''
###################################################################### 
paths35 = np.array([[testpath+'/images_noisy/blended_noisy{}.npy'.format(i) for i in sigmas[4]],
                   [testpath+'/images_noisy/not_blended_noisy{}.npy'.format(i) for i in sigmas[4]]])
#Getting the images 
blended_35 = [extract_test_images(paths35[0][j]) for j in range(5)]
not_blended_35 = [extract_test_images(paths35[1][j]) for j in range(5)]

####Run sep
for i,j in zip(range(5), sigmas[4]):
    sep_results(blends=blended_35[i], no_blends = not_blended_35[i], sigma_val=j, path=testpath)

'''6. $\sigma_{noise} = 40$'''
#####################################################################
paths40 = np.array([[testpath+'/images_noisy/blended_noisy{}.npy'.format(i) for i in sigmas[5]],
                   [testpath+'/images_noisy/not_blended_noisy{}.npy'.format(i) for i in sigmas[5]]])
#Getting the images 
blended_40 = [extract_test_images(paths40[0][j]) for j in range(5)]
not_blended_40 = [extract_test_images(paths40[1][j]) for j in range(5)]

####Run sep
for i,j in zip(range(5), sigmas[5]):
    sep_results(blends=blended_40[i], no_blends = not_blended_40[i], sigma_val=j, path=testpath)

'''REAL IMAGES '''
####################################################################
path_real = ['/Users/lacan/Documents/Cosmostat/Codes/BlendHunter/bh_real/blended_noisy.npy',
             '/Users/lacan/Documents/Cosmostat/Codes/BlendHunter/bh_real/not_blended_noisy.npy']

#Getting the images blended and not blended
blended_real = extract_test_images(path_real[0])
not_blended_real = extract_test_images(path_real[1])

#Run sep
sep_results(blends=blended_real, no_blends = not_blended_real, sigma_val='real', path=testpath)

'''Padded images '''
####################################################################
path_pad5 = ['/Users/lacan/Documents/Cosmostat/Codes/BlendHunter/bh_pad5/blended_noisy.npy',
             '/Users/lacan/Documents/Cosmostat/Codes/BlendHunter/bh_pad5/not_blended_noisy.npy']

#Getting the images blended and not blended
blended_pad5 = extract_test_images(path_pad5[0])
not_blended_pad5 = extract_test_images(path_pad5[1])

#Run sep
runner_pad5 = Run_Sep()
flags_b_pad5, sep_res_b_pad5 = runner5.process(blended_pad5)
flags_nb_pad5, sep_res_nb_pad5 = runner5.process(not_blended_pad5)
    
#Display results
acc_pad5= (len(np.where(flags_b_pad5 == 1)[0])+len(np.where(flags_nb_pad5 == 0)[0]))/(len(flags_b_pad5)+len(flags_nb_pad5))
print('Sep Accuracy (pad_35) : {}%'.format(acc_pad5*100))



'''Mixed noise in dataset '''
####################################################################
path_mn = ['/Users/lacan/Documents/Cosmostat/Codes/BlendHunter/bh_mix_close/blended_noisy.npy',
             '/Users/lacan/Documents/Cosmostat/Codes/BlendHunter/bh_mix_close/not_blended_noisy.npy']

#Getting the images blended and not blended
blended_mn = extract_test_images(path_mn[0])
not_blended_mn = extract_test_images(path_mn[1])

#Run sep
runner_mn = Run_Sep()
flags_b_mn, sep_res_b_mn = runner_mn.process(blended_mn)
flags_nb_mn, sep_res_nb_mn = runner_mn.process(not_blended_mn)
    
#Display results
acc_mn = (len(np.where(flags_b_mn == 1)[0])+len(np.where(flags_nb_mn == 0)[0]))/(len(flags_b_mn)+len(flags_nb_mn))
print('Sep Accuracy (mixed noise) : {}%'.format(acc_mn*100))


















