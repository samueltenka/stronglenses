''' author: sam tenka
    date: 2016-11-20
    descr: Read prepared data arrays
'''

from utils.config import get
import numpy as np

def fetch_Xy(class_nm):
    ''' Return Xy pair of given class
        ('VIS', 'TRAIN', or 'TEST')
    '''
    X, y = (np.load(get('DATA.%s.%s' % (class_nm, nm)))
            for nm in ('X', 'y'))
    return X, y
    
