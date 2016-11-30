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
from collections import Counter

def rescale_colors(Xs):
    ''' Perform channelwise affine transform into range [0, 1]. '''
    maxes = np.amax(Xs, axis=(0, 1, 2))
    mins = np.amin(Xs, axis=(0, 1, 2))
    for i in range(3):
        Xs[:,:,:,i] = (Xs[:,:,:,i]-mins[i]) / (maxes[i]-mins[i])
    return Xs

def get_stats(ys):
    print(ys)
    C = Counter(ys)
    for i in range(2):
        print('I see %d instances of label %d.' % (C[i], i))

def visualize():
    ''' Interactively display images of specified class.
        
        User enters '0', '1', or ''; a random instance of
        class 0, 1, or any class is then displayed. Close
        the popup window to continue, and exit via 'exit'.
    '''
    X_vis, y_vis = fetch_Xy('VIS') 
    X_vis = rescale_colors(X_vis)
    get_stats(y_vis)
    for label_nm in user_input_iterator():
        while True:
            index = randrange(len(X_vis))
            img, label = X_vis[index], y_vis[index]
            if (label_nm not in ('0', '1')) or (label == int(label_nm)): break
        print('Image %d has label %d' % (index, label))
        plt.title('Image %d has label %d' % (index, label))
        plt.imshow(img) 
        plt.show()

if __name__=='__main__':
    visualize()
