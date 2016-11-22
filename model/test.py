''' author: sam tenka
    credits: 
    date: 2016-11-16
    descr: Test neural net. 
    usage:
        from model.test import nntest
'''

from __future__ import print_function

def nntest(model, Xy_test, batch_size=32):
    X_test, y_test = Xy_test
    loss, acc = model.evaluate(X_test, y_test, batch_size=batch_size)
    return loss, acc
 
