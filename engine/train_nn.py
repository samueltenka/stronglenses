''' author: sam tenka
    credits: 
    date: 2016-11-16
    descr: Identify strong lenses via neural net 
    usage:
        python -m engine.train_nn
'''

from __future__ import print_function
from utils.config import get
from keras.models import load_model
from model.model import make_MLP
from model.train import nntrain
from model.test import nntest
from utils.terminal import print_boxed
from os.path import isfile
import numpy as np

from data_scrape.fetch_data import fetch_Xy

def get_model(checkpoint):
    if isfile(checkpoint):
        print('Loading from %s...' % checkpoint)
        return load_model(checkpoint)
    return make_MLP()

def train_then_test(model, Xy_test, Xy_train, checkpoint=None, nb_epoch=20):
    nntrain(model, Xy_train, checkpoint=checkpoint, nb_epoch=nb_epoch)
    loss, acc = nntest(model, Xy_test)
    print_boxed('loss=%.4f, acc=%.4f' % (loss, acc))

checkpoint = get('MODEL.CHECKPOINT')
Xy_test, Xy_train = fetch_Xy('TEST'), fetch_Xy('TRAIN')
model = get_model(checkpoint)
train_then_test(model, Xy_test, Xy_train, checkpoint)
