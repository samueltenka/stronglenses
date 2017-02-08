''' author: daniel zhang
    date: 2016-11-19
    descr: Obtain and prepare dataset:
                Download dataset from (dropbox) link supplied in config.json
                Prepare dataset by:
                    converting to tensorflow indexing and
                    splitting into vis(ualize) / train / test.
    usage: Run the following command:
        python -m data_scripts.download
'''
import numpy as np
from sklearn.cross_validation import train_test_split
from utils.config import get
import os

def download_arrays():
    ''' Download dataset from (dropbox) link supplied in config.json.
        
    '''
    os.system('wget -O %s %s' % (get('DATA.FULL.X'), get('DATA.LINKS.X')))
    os.system('wget -O %s %s' % (get('DATA.FULL.y'), get('DATA.LINKS.y')))

def load_arrays():
    ''' '''
    X = np.load(get('DATA.FULL.X'))
    y = np.load(get('DATA.FULL.y'))
    return (X, y)

def to_tensorflow(X):
    ''' Prepare data for use with tensorflow. Specifically, transform array
        indexed channels-initial to one indexed channels-final. '''
    return [x.transpose((1, 2, 0)) for x in X]

def split_3(X, y):
    ''' Split given dataset into 3 parts: vis(ualize) / train / test. Note: if
        we imagine a continuum of usage frequency that spans between pure
        training sets to pure test sets, then a complete description of our
        work would have have further sets: 
            vis --- train --- val --- test --- fresh test --- physical test   
            ^       ^                 ^
        The marked sets are those produced here. The other sets appear as
        follows: we base model-checkpointing on scores on a val(idation) set
        split by Keras from the trainset; to counter overfitting on the
        multiply-used "testset", we may sample a fresh testset from the
        simulator to compare competitive models; and we hope that we generalize
        from the simulator-trained models to real, physical data --- the truest
        of test sets. 
    '''
    X_use, X_vis, y_use, y_vis = train_test_split(X, y,
        test_size=get('DATA.SPLIT.VIS_VS_USE'), random_state=get('DATA.SPLIT.SEED_A'))
    X_train, X_test, y_train, y_test = train_test_split(X_use, y_use,
       test_size=get('DATA.SPLIT.TEST_VS_TRAIN'), random_state=get('DATA.SPLIT.SEED_B'))
    return {'VIS':(X_vis, y_vis),
            'TRAIN':(X_train, y_train),
            'TEST':(X_test, y_test)}

def save_array(Xy, class_nm):
    ''' '''
    X, y = Xy
    for nm, array in {'X':X, 'y':y}.items():
        np.save(get('DATA.%s.%s'%(class_nm, nm)), array)

if __name__=='__main__':
    ''' Putting it all together ... '''
    download_arrays()
    X, y = load_arrays()
    X = to_tensorflow(X)
    Xys_by_class_nm = split_3(X, y)
    for class_nm, Xy in Xys_by_class_nm.items():
        save_array(Xy, class_nm) 
