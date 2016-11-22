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

def view_history():
    ''' Interactively display images of specified class.
        
        User enters '0', '1', or ''; a random instance of
        class 0, 1, or any class is then displayed. Close
        the popup window to continue, and exit via 'exit'.
    '''
    for model_nm in user_input_iterator():
        history_nm = get('MODEL.%s.HISTORY' % model_nm)
        with open(history_nm) as f:
            history = eval(f.read())
        plt.plot(history['loss'], label='train loss')
        plt.plot(history['val_loss'], label='val loss')
        plt.title('Training History of %s' % (model_nm))
        plt.legend()
        plt.show()

if __name__=='__main__':
    view_history()
