''' author: sam tenka
    date: 2016-11-21
    descr: Compute and view yield curves (on test-set).
    usage: Type
            python -m engine.view_curves
        Type 'yield' or 'conf' to switch between modes.
        Type a space-separated list of names (e.g. 'MLP SHALLOW_RES')
        to produce plots. 
'''

from __future__ import print_function
from utils.config import get
from utils.terminal import user_input_iterator
from utils.algo import memoize
from data_scrape.fetch_data import fetch_Xy
from model.fetch_model import fetch_model 
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import fftconvolve

X, y = fetch_Xy('TEST')

def blur(array, sigma=1.0, N=6):
    L = int(1+N*sigma)
    gaussian = np.sqrt(1.0 / (2*np.pi * sigma**2)) * np.exp(- np.square(np.arange(-L, L+1)) / (2 * sigma**2))
    return fftconvolve(array, gaussian)[L:-L]

@memoize
def get_cums(model_nm, bins=100, eps=1e-4):
    ''' Return domain [0.5, 1.0] of conf, cumulative counts of correct predictions
        on sets of given min conf, and cumulative counts of all predictions on
        those sets.
    '''
    model, _ = fetch_model(model_nm)
    predictions = model.predict(X, batch_size=30, verbose=1)[:,0]
    confidences = np.arange(0.5, 1.0 + 1.0/(2*bins), 1.0/(2*bins))
    correct_histogram = np.zeros(bins + 1) + eps
    total_histogram = np.zeros(bins + 1) + eps
    for p, label in zip(predictions, y):
        c = abs(p-0.5)
        index = int(c*bins) 
        total_histogram[index] += 1.0
        if round(p) != label: continue
        correct_histogram[index] += 1.0
    correct_cum = np.cumsum(correct_histogram[::-1])[::-1]
    total_cum = np.cumsum(total_histogram[::-1])[::-1]
    correct_cum = blur(correct_cum, sigma=bins/40)
    total_cum = blur(total_cum, sigma=bins/40)
    return confidences, correct_cum, total_cum 

def conf_curve(model_nm):
    ''' A `confidence curve` plots accuracy vs min conf '''
    confs, correct_cum, total_cum = get_cums(model_nm)
    return confs, (correct_cum/total_cum)

def yield_curve(model_nm):
    ''' A `yield curve` plots accuracy vs yield.

        The blurring done in `get_cums` potentially makes
        `total_cum` non-monotonic; we remedy this by chopping
        at the argmax.
    '''
    confs, correct_cum, total_cum = get_cums(model_nm)
    i = np.argmax(total_cum)
    return (total_cum/max(total_cum))[i:], (correct_cum/total_cum)[i:]

def compute_curve(model_nm, mode): 
    return {'yield':yield_curve,
            'conf':conf_curve
            }[mode](model_nm)

def view_curves():
    ''' Interactively save and display yield or confidence curves
        of specified model.

        TODO: merge with history viewing, and refactor all
    '''
    mode = 'yield'  
    for command in user_input_iterator():
        if command in ('yield', 'conf'):
            mode = command
            print('switched to `%s` mode' % mode)
            continue
        nms = command.split(' ')
        for nm in nms:
            color = get('MODEL.%s.PLOTCOLOR' % nm)
            domain, accuracies = compute_curve(nm, mode)
            plt.plot(domain, accuracies, label='%s' % nm,
                     color=color, ls='-', lw=2.0)
        if mode=='conf':
            plt.plot(domain, domain, label='truth',
                         color='k', ls='-', lw=1.0)
            plt.title('Confidence curves of %s' % ', '.join(nms))
            plt.gca().set_xlabel('Confidence threshold C')
            plt.gca().set_ylabel('Accuracy on data on which model is at least C confident')
        else:
            plt.title('Yield curves of %s' % ', '.join(nms))
            plt.gca().set_xlabel('Fraction F of data')
            plt.gca().set_ylabel('Accuracy on top F of data, by confidence')
        plt.legend(loc='best')
        plt.show()

if __name__=='__main__':
    view_curves()
