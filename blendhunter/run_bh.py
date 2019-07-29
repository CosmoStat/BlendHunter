import numpy as np
import sys

bh_path = ('/home/alice/Cosmostat/Codes/BlendHunter')
sys.path.extend([bh_path])

# Set plaidml backend for Keras before importing blendhunter
import plaidml.keras
plaidml.keras.install_backend()

from blendhunter import BlendHunter


path = '/home/alice/Cosmostat/Codes/BlendHunter/bh'

bh = BlendHunter(weights_path=path + '/weights')

# Train Network
bh.train(path + '/BlendHunterData',
         get_features=True,
         train_top=True,
         fine_tune=False)

# Predict Results
pred_top = bh.predict(path + '/BlendHunterData/test/test', weights_type='top')
pred_fine = bh.predict(path + '/BlendHunterData/test/test',
                       weights_type='fine')
true = np.load(path + '/BlendHunterData/test/test/labels.npy')
print("Match Top:", np.sum(pred_top == true) / true.size)
print("Match Fine:", np.sum(pred_fine == true) / true.size)

print("Error Top", np.sum(pred_top != true) / true.size)

#Save results
np.save('/home/alice/Cosmostat/Codes/BlendHunter/bh/BlendHunterData/test/test/pred_top_tune.npy', pred_top)
np.save('/home/alice/Cosmostat/Codes/BlendHunter/bh/BlendHunterData/test/test/pred_finetune.npy', pred_fine)



# Plot and label comparison
