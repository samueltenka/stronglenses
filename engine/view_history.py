''' author: sam tenka
    date: 2016-11-21
    descr: View training histories.
'''

from __future__ import print_function
from utils.config import get
from utils.terminal import user_input_iterator
from data_scrape.fetch_data import fetch_Xy
import numpy as np
import matplotlib.pyplot as plt

def get_history(model_nm):
    history_nm = get('MODEL.%s.HISTORY' % model_nm)
    with open(history_nm) as f:
        return eval(f.read())

def view_history():
    ''' Interactively display training plots of specified model.

        The user enters a space-separated list of model names,
        such as 'MLP SHALLOW_RES'. Overfitting and convergence
        within a model, as well as loss advantages between models,
        are then apparent from the plot.

        TODO: display minimum as horizontal line; display history
        cuts as vertical lines; compute worstcase accuracy given loss.
    '''
    for model_nms in user_input_iterator():
        nms = model_nms.split(' ')
        for nm, color in zip(nms, 'rgbcmy'):
            history = get_history(nm)
            plt.plot(history['loss'], label='%s train loss' % nm,
                     color=color, ls='-', lw=1.0)
            plt.plot(history['val_loss'], label='%s val loss' % nm,
                     color=color, ls='-', lw=3.0)
            min_val_loss = min(history['val_loss'])
        plt.title('Training History of %s' % ','.join(nms))
        plt.legend()
        plt.show()

if __name__=='__main__':
    view_history()
