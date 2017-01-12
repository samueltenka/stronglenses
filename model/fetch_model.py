''' author: sam tenka
    credits:
    date: 2016-11-21
    descr: Fetch model. 
    usage:
'''

from __future__ import print_function
from utils.config import get
from model.make_model import make_logistic, make_mlp, make_mlp_wide, make_mlp_l2, \
                             make_shallow_res, make_res_2, make_squeeze_res, make_squeeze_res_wide, \
                             make_squeeze_skip, make_softplus_3
from keras.models import load_model
from os.path import isfile

models_by_name = {
    'LOGISTIC': make_logistic,
    'MLP': make_mlp,
    'MLP_WIDE': make_mlp_wide,
    'MLP_L2': make_mlp_l2,
    'SHALLOW_RES': make_shallow_res,
    'RES_2': make_res_2,
    'SQUEEZE_RES': make_squeeze_res,
    'SQUEEZE_RES_WIDE': make_squeeze_res_wide,
    'SOFTPLUS_3': make_softplus_3,
    'SQUEEZE_SKIP': make_squeeze_skip,
}
def fetch_model(model_nm):
    ''' Return (model, checkpoint name) pair. '''
    checkpoint = get('MODEL.%s.CHECKPOINT' % model_nm)
    if isfile(checkpoint):
        print('Loading from %s...' % checkpoint)
        return load_model(checkpoint), checkpoint
    return models_by_name[model_nm](), checkpoint
