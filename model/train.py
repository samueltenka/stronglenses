''' author: sam tenka
    credits:
        We learned how to checkpoint in Keras from:  
            http://machinelearningmastery.com/check-point-deep-learning-models-keras/
    date: 2016-11-16
    descr: Train neural net.
    usage:
        from model.train import nntrain
'''

from __future__ import print_function
from keras.callbacks import ModelCheckpoint

def nntrain(model, Xy_train, validation_split=0.1,
            checkpoint=None, nb_epoch=20, batch_size=32):
    ''' Train `model` for `nb_epoch` epochs or until keyboard interrupt;
        eithe way, save to `checkpoint`.
  
        Arguments: 
            validation_split: proportion of trainset to use not for backprop,
                              but instead for monitoring loss in pocket algorithm.
            checkpoint:       filename to which to save model checkpoints.
    '''
    callbacks = [] if not checkpoint else \
                [ModelCheckpoint(checkpoint, monitor='val_loss', save_best_only=True)]
    X_train, y_train = Xy_train
    try:
        for i in range(nb_epoch):
            model.fit(X_train, y_train, validation_split=validation_split,
                      callbacks=callbacks, nb_epoch=1, batch_size=batch_size,
                      verbose=1)
    except KeyboardInterrupt:
        print()
        pass
