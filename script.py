# -*- coding: utf-8 -*-

from blendhunter.network import VGG16

train_path = '/Users/sfarrens/Desktop/test_blend/train'
valid_path = '/Users/sfarrens/Desktop/test_blend/valid'


deep = VGG16(train_path, valid_path, batch_size=1, epochs=5)

deep.train_network()
