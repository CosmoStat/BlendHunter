import numpy as np
import sys

bh_path = ('/Users/lacan/Documents/Cosmostat/Codes/BlendHunter')
sys.path.extend([bh_path])

# Set plaidml backend for Keras before importing blendhunter
import plaidml.keras
plaidml.keras.install_backend()

from blendhunter import BlendHunter
from os.path import expanduser
user_home = expanduser("~")


for j in [5, 14, 18, 26, 35, 40]:
    for i in [1,2,3,4]:
        #Make sure to check the folder hierarchy
        path = user_home+'/Documents/Cosmostat/Codes/BlendHunter/bh_{}'.format(str(j)+str(i))
        bh = BlendHunter(weights_path=path + '/weights')

        # Train Network

        bh.train(path + '/BlendHunterData',
         get_features=True,
         train_top=True,
         fine_tune=False)

        hist = np.array(bh.history.history)

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
        np.save(user_home+'/Documents/Cosmostat/Codes/BlendHunter/pad_results/preds_pad{}.npy'.format(str(j)+str(i)), pred_top)
