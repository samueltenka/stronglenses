''' author: sam tenka
    credits:
    date: 2016-11-21
    descr: Fetch model. 
    usage:
'''

from __future__ import print_function
from utils.config import get
from model.make_model import make_mlp, make_mlp_wide, make_shallow_res, make_res_2, make_squeeze_res, make_softplus_3
from keras.models import load_model
from os.path import isfile

models_by_name = {
    'MLP': make_mlp,
    'MLP_WIDE': make_mlp_wide,
    'SHALLOW_RES': make_shallow_res,
    'RES_2': make_res_2,
    'SQUEEZE_RES': make_squeeze_res,
    'SOFTPLUS_3': make_softplus_3
}
def fetch_model(model_nm):
    ''' Return (model, checkpoint name) pair. '''
    checkpoint = get('MODEL.%s.CHECKPOINT' % model_nm)
    if isfile(checkpoint):
        print('Loading from %s...' % checkpoint)
        return load_model(checkpoint), checkpoint
    return models_by_name[model_nm](), checkpoint
