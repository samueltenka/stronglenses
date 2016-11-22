''' author: sam tenka
    credits: 
    date: 2016-11-16
    descr: Identify strong lenses via neural net 
    usage:
        python -m engine.train_nn
'''

from __future__ import print_function
from utils.config import get
from utils.terminal import print_boxed
from data_scrape.fetch_data import fetch_Xy
from model.fetch_model import fetch_model
from model.train import nntrain
from model.test import nntest

def train_then_test(model, Xy_test, Xy_train, checkpoint=None, nb_epoch=20):
    nntrain(model, Xy_train, checkpoint=checkpoint, nb_epoch=nb_epoch)
    loss, acc = nntest(model, Xy_test)
    print_boxed('loss=%.4f, acc=%.4f' % (loss, acc))

model_nm = get('MODEL.CURRENT') 
Xy_test, Xy_train = fetch_Xy('TEST'), fetch_Xy('TRAIN')
model, checkpoint = fetch_model(model_nm)
train_then_test(model, Xy_test, Xy_train, checkpoint)

