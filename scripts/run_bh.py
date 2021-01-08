import numpy as np
import sys
from os.path import expanduser
user_home = expanduser("~")

#Check the folder hierarchy
bh_path = (user_home+'/Cosmostat/Codes/BlendHunter')
sys.path.extend([bh_path])

"""Set plaidml backend for Keras before importing blendhunter"""
import plaidml.keras
plaidml.keras.install_backend()

#Import network
from blendhunter import BlendHunter

"""Loop to train the network on padded noisy images and get predicted labels"""
for j in [5, 14, 18, 26, 35, 40]:
    for i in ['', 1,2,3,4]:
        #Make sure to check the folder hierarchy
        path = user_home+'/Cosmostat/Codes/BlendHunter/bh_{}'.format(str(j)+str(i))
        #For non padded images
        #path = user_home+'/Cosmostat/Codes/BlendHunter/bh_{}'.format(str(j)+str(i))
        """Make sure each bh folder contains a 'weights' folder"""
        bh = BlendHunter(weights_path=path + '/weights')

        # Train Network
        bh.train(path + '/BlendHunterData',
         get_features=True,
         train_top=True,
         fine_tune=False) #No fine tuning

        hist = np.array(bh.history.history) #Saving training history is optional

        # Predict Results
        pred_top = bh.predict(path + '/BlendHunterData/test/test', weights_type='top')
        #pred_fine = bh.predict(path + '/BlendHunterData/test/test', weights_type='fine')
        true = np.load(path + '/BlendHunterData/test/test/labels.npy')
        print("Match Top:", np.sum(pred_top == true) / true.size)
        #print("Match Fine:", np.sum(pred_fine == true) / true.size)
        print("Error Top", np.sum(pred_top != true) / true.size)

        #Save history and results
        #Make sure to check the folder hierarchy
        np.save(path+'/BlendHunterData/test/test/history.npy', hist)
        np.save(user_home+'/Cosmostat/Codes/BlendHunter/bh_results/preds{}.npy'.format(str(j)+str(i)), pred_top)
        #For non padded images
        #np.save(user_home+'/Cosmostat/Codes/BlendHunter/bh_results/preds_{}.npy'.format(str(j)+str(i)), pred_top)
