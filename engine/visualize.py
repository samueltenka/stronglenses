''' author: sam tenka
    date: 2016-11-20
    descr: Visualize vis-set images.
'''

from __future__ import print_function
from utils.config import get
from utils.terminal import user_input_iterator
from data_scrape.fetch_data import fetch_Xy
import numpy as np
import matplotlib.pyplot as plt
from random import randrange

def rescale_colors(Xs):
    ''' Channelwise affine transform into range [0, 1]. '''
    maxes = np.amax(Xs, axis=(0, 1, 2))
    mins = np.amin(Xs, axis=(0, 1, 2))
    for i in range(3):
        Xs[:,:,:,i] = (Xs[:,:,:,i]-mins[i]) / (maxes[i]-mins[i])
    return Xs

def visualize():
    X_vis, y_vis = fetch_Xy('VIS') 
    X_vis = rescale_colors(X_vis)
    for command in user_input_iterator():
        while True:
            index = randrange(len(X_vis))
            img, label = X_vis[index], y_vis[index]
            if (not command) or (label == int(command)): break
        print('label: %d (index %d)' % (label, index))
        plt.imshow(img) 
        plt.show()
