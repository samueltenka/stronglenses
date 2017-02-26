''' author: sam tenka
    date: 2016-11-20
    descr: Visualize vis-set images.
    usage: Type the command
        python -m engine.visualize
    An interactive prompt should appear, displaying class statistics of the
    visualization set. Enter an empty string to display an image uniformly
    sampled from the vis-set. A command-line message should reveal the sample's
    label. The vis-set is affine-normalized in each channel to be representable
    without loss within RGB color space. Enter a '0' or a '1' to display an
    image uniformly sampled from the respective class within the vis-set. Enter
    a string of '0's and '1's to display mutiple such images side-by-side.
    Within this string, '?'s are permitted: they denote images samples from the
    whole set. For example, entering '0?1' would display 3 images, the middle
    one of randomly chosen class. Enter 'exit' to exit.
'''

from __future__ import print_function
from utils.config import get
from utils.terminal import user_input_iterator
from data_scripts.access import fetch_Xy
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
    ''' Print class statistics of a given integer array of labels. '''
    print(ys)
    C = Counter(ys)
    for i in range(2):
        print('I see %d instances of label %d.' % (C[i], i))

def sample_from_class(label_nm, X_vis, y_vis):
    ''' Return a uniform random sample from a specified subset of the provided
        visualization set. Return value is a tuple
            (selected index in vis set, image as numpy array, integer label)
        `label_nm` is a string: if it is '0' or '1', then the sample is taken
        from the corresponding class. Else, the sample is taken from the whole
        vis set. While not enforced, it is encouraged to use character '?' for
        that latter case. 
        
        Naive implementation remains fast on small visualization sets. '''
    while True:
        index = randrange(len(X_vis))
        img, label = X_vis[index], y_vis[index]
        if (label_nm not in ('0', '1')) or (label == int(label_nm)):
            return index, img, label

def visualize():
    ''' Interactively display images of specified class.
        
        User enters '0', '1', or ''; a random instance of
        class 0, 1, or any class is then displayed. Close
        the popup window to continue, and exit via 'exit'.

        See file-level documentation for details.
    '''
    X_vis, y_vis = fetch_Xy('VIS') 
    X_vis = rescale_colors(X_vis)
    get_stats(y_vis)
    for label_nms in user_input_iterator():
        if not label_nms: label_nms = '?'
        fig = plt.figure()
        for i, label_nm in enumerate(label_nms):
            index, img, label = sample_from_class(label_nm, X_vis, y_vis)
            axis = fig.add_subplot(1, len(label_nms), i+1)
            axis.set_title('Image %d has label %s' % (index, label_nm))
            print('Image %d has label %d' % (index, label))
            plt.xticks([])
            plt.yticks([])
            plt.imshow(img) 
        plt.show()

if __name__=='__main__':
    visualize()
