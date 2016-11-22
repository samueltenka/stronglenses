''' author: sam tenka
    credits:
    date: 2016-11-21
    descr: Fetch model. 
    usage:
'''

from __future__ import print_function
from utils.config import get
from model.make_model import make_mlp, make_mlp_wide, make_shallow_res
from keras.models import load_model
from os.path import isfile

models_by_name = {
    'MLP': make_mlp,
    'MLP_WIDE': make_mlp_wide,
    'SHALLOW_RES': make_shallow_res
}
def fetch_model(model_nm):
    checkpoint = get('MODEL.%s.CHECKPOINT' % model_nm)
    if isfile(checkpoint):
        print('Loading from %s...' % checkpoint)
        return load_model(checkpoint), checkpoint
    return models_by_name[model_nm](), checkpoint
