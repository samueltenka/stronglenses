''' author: daniel zhang
    date: 2016-11-19
    descr: Prepare dataset by converting to tensorflow indexing and
           splitting into three parts: vis(ualize) / train / test.
'''
import numpy
from sklearn.cross_validation import train_test_split
from utils.config import get

def load_arrays():
    X = numpy.load(get('DATA.FULL.X')))
    y = numpy.load(get('DATA.FULL.y')))
    return (X, y)

def to_tensorflow(X):
    ''' Transform array indexed channels-initial
        to one indexed channels-final. '''
    return [x.transpose((1, 2, 0)) for x in X]

def split_3(X, y):
    X_use, X_vis, y_use, y_vis = train_test_split(X, y,
        test_size=get('DATA.SPLIT.VIS_VS_USE'), random_state=get('DATA.SPLIT.SEED_A'))
    X_train, X_test, y_train, y_test = train_test_split(X_use, Y_use,
       test_size=get('DATA.SPLIT.TEST_VS_TRAIN'), random_state=get('DATA.SPLIT.SEED_B'))
    return {'vis':(X_vis, y_vis),
            'train':(X_train, y_train),
            'test':(X_test, y_test)}

def save_array(Xy, class_nm):
    X, y = Xy
    numpy.save(config['DATA.%s.X'%part], X)
    numpy.save(config['DATA.%s.y'%part], y)

if __name__=='__main__':
    ''' Putting it all together ... '''
    X, y = load_arrays()
    X = to_tensorflow(X)
    Xys_by_class_nm = split_3(X, y)
    for class_nm, Xy in Xys_by_class_nm.items():
        save_array(Xy, class_nm) 
