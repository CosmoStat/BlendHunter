# -*- coding: utf-8 -*-

from blendhunter import BlendHunter


path = ('$HOME/Documents/Codes/tutorial/keras/data/kaggle_cats_vs_dogs/test/'
        'test_images')

net = BlendHunter(image_shape=(150, 150, 3), classes=('cats', 'dogs'))
res = net.predict(path)

print(res)
