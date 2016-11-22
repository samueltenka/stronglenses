''' author: sam tenka
    date: 2016-11-21
    descr: Compute and view yield curves (on vis-set).
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
    gaussian = np.exp(- np.square(np.arange(-L, L+1)) / (2 * sigma**2))
    return fftconvolve(array, gaussian)[L:-L]

@memoize
def compute_confidence_curve(model_nm, bins=100, eps=1e-4):
    ''' A `yield curve` is a plot of   Returns an array [] '''
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
    correct_cum = blur(np.cumsum(correct_histogram[::-1])[::-1], sigma=bins/40)
    total_cum = blur(np.cumsum(total_histogram[::-1])[::-1], sigma=bins/40)
    return confidences, (correct_cum/total_cum)



def view_confidence_curves():
    ''' Interactively save and display yield curves of specified model.
    '''
    for model_nms in user_input_iterator():
        nms = model_nms.split(' ')
        for nm in nms:
            color = get('MODEL.%s.PLOTCOLOR' % nm)
            confidences, accuracies = compute_confidence_curve(nm)
            plt.plot(confidences, accuracies, label='%s' % nm,
                     color=color, ls='-', lw=2.0)
        plt.plot(confidences, confidences, label='truth',
                     color='k', ls='-', lw=1.0)
        plt.title('Confidence curves of %s' % ', '.join(nms))
        plt.gca().set_xlabel('Model\'s confidence in prediction')
        plt.gca().set_ylabel('Accuracy on points deemed at most that confident.')
        plt.legend(loc='best')
        plt.show()

if __name__=='__main__':
    view_confidence_curves()
