''' author: sam tenka
    date: 2016-oct-13
    descr: read numpy arrays for training and testing
'''

import numpy as np

with open('engine/engine.config') as f:
    config = eval(f.read())

X = np.load(config['DATA']['X_path'])
y = np.load(config['DATA']['y_path'])

