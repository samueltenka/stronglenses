''' author: sam tenka
    credits: 
    date: 2016-11-16
    descr: Identify strong lenses via neural net 
    usage: Try: 
            python -m engine.train_nn [NB_EPOCH]
        NB_EPOCH has a default value set in the
        config file. If NB_EPOCH is 0, this script
        just tests the neural net without training.
'''

from __future__ import print_function
from utils.config import get
from utils.terminal import print_boxed
from data_scrape.fetch_data import fetch_Xy
from model.fetch_model import fetch_model
from model.train import nntrain
from model.test import nntest
from os.path import isfile
from sys import argv
import numpy as np

history_keys = ('loss', 'acc', 'val_loss', 'val_acc')
blank_history = {k:[] for k in history_keys}

def train_then_test(model, Xy_test, Xy_train, checkpoint, nb_epoch):
    ''' Train and test a given neural net; display model performance,
        manage checkpoints, and return history. 

        Keras is inconsistent in that training after 0 epochs doesn't
        even return a blank history; this inconsistency we explicitly remedy.
    '''
    history = nntrain(model, Xy_train, checkpoint=checkpoint, nb_epoch=nb_epoch)
    history = history.history if nb_epoch else blank_history
    loss, acc = nntest(model, Xy_test)
    print_boxed('loss=%.4f, acc=%.4f' % (loss, acc))
    return history

def update_history_log(log_nm, history):
    ''' Append history to history file. Create new file if none extant. 

        History is written as the string representation of a Python
        dictionary of lists-by-strings, as in `blank_history` above. 
    '''
    if isfile(log_nm): 
        with open(log_nm) as f:
            old = eval(f.read())
            history = {k: old[k]+v for k, v in history.items()} 
    with open(log_nm, 'w') as f:
        f.write(str(history)) 

def nb_epoch_from_stdin():
    ''' Return nb_epoch, as specified in user command or config file. '''
    return get('TRAIN.NB_EPOCH') if not argv[1:] else int(argv[1:][0])

def run(): 
    ''' Train, test, then update checkpoints, history, predictions. '''
    # 0. Fetch data and model
    nb_epoch = nb_epoch_from_stdin()
    model_nm = get('MODEL.CURRENT') 
    Xy_test, Xy_train = fetch_Xy('TEST'), fetch_Xy('TRAIN')
    model, checkpoint = fetch_model(model_nm)

    # 1. Train, test, write checkpoint and history
    history = train_then_test(model, Xy_test, Xy_train, checkpoint, nb_epoch)
    log_nm = get('MODEL.%s.HISTORY' % model_nm)
    update_history_log(log_nm, history)

    # 2. Write predictions 
    X, y = Xy_test
    preds = model.predict(X, batch_size=32, verbose=0)[:,0]
    with open(get('MODEL.%s.PREDICTION' % model_nm), 'w') as f:
        np.save(f, preds)

if __name__=='__main__':
    run()
