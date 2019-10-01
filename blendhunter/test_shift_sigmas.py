import numpy as np
import sys

bh_path = ('/Users/alacan/Documents/Cosmostat/Codes/BlendHunter')
sys.path.extend([bh_path])

from blendhunter import BlendHunter

#Predict using given pretrained weights
def run_bh(path, value):

    bh = BlendHunter(weights_path=path + '/weights{}'.format(value))
    # Train Network
    bh.train(path + '/BlendHunterData',
         get_features=False,
         train_top=False,
         fine_tune=False)

    # Predict Results
    pred_top = bh.predict(path + '/BlendHunterData/test/test', weights_type='top')
    true = np.load(path + '/BlendHunterData/test/test/labels.npy')
    print("Match Top:", np.sum(pred_top == true) / true.size)
    print("Error Top", np.sum(pred_top != true) / true.size)

    return np.sum(pred_top == true) / true.size

#Save results to dictionary
def get_results(paths_list=None, n_path=None, sigma_value=None):

    results_dict = dict()

    for i,j in zip(paths_list, n_path):
        acc = run_bh(i, sigma_value)

        #Save results
        results_dict.update({'Path'+str(j):{'Accuracy': acc}})

    return results_dict

path_bh = '/Users/alacan/Documents/Cosmostat/Codes/BlendHunter'
path_shift =path_bh + '/more_noise_ranges'
path3 = path_shift+'/bh_3'
path5 = path_bh+'/bh_5'
path7 = path_shift+'/bh_7'
path10 = path_shift+'/bh_10'
path12 = path_shift+'/bh_12'
path14 = path_bh+'/bh_14'
path16 = path_shift+'/bh_16'
path18 = path_bh+'/bh_18'
path20 = path_shift+'/bh_20'
path22 = path_bh+'/bh_22'
path24 = path_shift+'/bh_24'
path26 = path_bh+'/bh_26'
path28 = path_shift+'/bh_28'
path30 = path_shift+'/bh_30'
path32 = path_shift+'/bh_32'
path35 = path_bh+'/bh_35'
path37 = path_shift+'/bh_37'
path40 = path_bh+'/bh_40'
path42 = path_shift+'/bh_42'
path44 = path_shift+'/bh_44'

paths = [path3, path5, path7, path10, path12, path14, path16, path18,path20, path22, path24, path26, path28, path30, path32, path35, path37, path40, path42, path44]
nb_path =[3,5,7,10, 12, 14, 16,18,20,22,24,26,28,30,32,35,37,40,42, 44]

#Retrieve results for each set of weights
results_weights40_3 = get_results(paths_list=paths, n_path=nb_path, sigma_value=403)

#Save dictionary
np.save('/Users/alacan/Documents/Cosmostat/Codes/BlendHunter/acc_weights40_3.npy', results_weights40_3)
