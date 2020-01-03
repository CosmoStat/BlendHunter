import numpy as np
import sys

bh_path = ('/Users/alacan/Documents/Cosmostat/Codes/BlendHunter')
sys.path.extend([bh_path])

# Set plaidml backend for Keras before importing blendhunter
import plaidml.keras
plaidml.keras.install_backend()

from blendhunter import BlendHunter
from os.path import expanduser
user_home = expanduser("~")

#Make sure to check the folder hierarchy
path = user_home+'/Documents/Cosmostat/Codes/BlendHunter/bh_'

bh = BlendHunter(weights_path=path + '/weights')

# Train Network
bh.train(path + '/BlendHunterData',
         get_features=False,
         train_top=False,
         fine_tune=False)
#hist = np.array(bh.history.history)

# Predict Results
pred_top = bh.predict(path + '/BlendHunterData/test/test', weights_type='top')
#pred_fine = bh.predict(path + '/BlendHunterData/test/test', weights_type='fine')
true = np.load(path + '/BlendHunterData/test/test/labels.npy')
print("Match Top:", np.sum(pred_top == true) / true.size)
#print(bh.history.history.keys())
#print("Match Fine:", np.sum(pred_fine == true) / true.size)
print("Error Top", np.sum(pred_top != true) / true.size)

#Save results
#np.save(path+'/BlendHunterData/test/test/history.npy', hist)
np.save(path+'/BlendHunterData/test/test/pred.npy', pred_top)
#np.save('/Users/alacan/Documents/Cosmostat/Codes/BlendHunter/bh/BlendHunterData/test/test/pred_finetune.npy', pred_fine)
