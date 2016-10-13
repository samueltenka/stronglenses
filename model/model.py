''' author: sam tenka
    credits: I am grateful for the following topical resources: 
        Convolutional Layers in Keras:
            https://keras.io/layers/convolutional/
        VGG16 archictecture:
            https://gist.github.com/baraldilorenzo/07d7802847aaad0a35d3
        Loc architecture:
            http://cs231n.stanford.edu/reports/cpd_final.pdf
    date: 2016-oct-12
    descr: Image-Classification Architectures for Strong Lenses.
        date: 2016-oct-06
    usage: try one of the following:
        from model.model import make_VGG16
        from model.model import make_MLP
        from model.model import make_Loc
'''

from __future__ import print_function

from keras.models import Sequential 
from keras.layers import Flatten, Dense, Activation, Dropout
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.layers.local import LocallyConnected2D

def add_layer(model, channels, kernel, layer_type, border_mode='same'):
    ''' Add a single kernel-type layer to `model`
    '''
    model.add(layer_type(channels, kernel, kernel,
                         border_mode=border_mode))
    model.add(Activation('relu'))

def add_block(model, depth, channels, kernel, pool, dropout=0.25, layer_type=Convolution2D, border_mode='same'):
    ''' Add a stack of kernel-type layers to `model`

        Afterward apply pooling and dropout-regularization.
    '''
    for i in range(depth):
        add_layer(model, channels, kernel, layer_type, border_mode)
    if pool!=1: model.add(MaxPooling2D((pool, pool)))
    if dropout: model.add(Dropout(dropout))

def add_convblock(model, depth, channels, kernel, pool, dropout=0.25):
    add_block(model, depth, channels, kernel, pool, dropout, layer_type=Convolution2D, border_mode='same')
def add_loclblock(model, depth, channels, kernel, pool, dropout=0.25):
    add_block(model, depth, channels, kernel, pool, dropout, layer_type=LocallyConnected2D, border_mode='valid')

def add_dense_classifier(model, hidden_sizes=[1024, 1024]):
    ''' Add a stack of fully-connected layers to `model` 
    '''
    model.add(Flatten())
    for hs in hidden_sizes:
        model.add(Dense(hs))
        model.add(Dropout(0.5))
        model.add(Activation('relu'))
    model.add(Dense(1, activation='sigmoid'))

def compile_classifier(make_model):
    ''' Decorate `make_model` to compile and summarize model.
    '''
    def compiler(*args, **kwargs):
        model = make_model(*args, **kwargs)
        model.compile(loss='binary_crossentropy',
                      optimizer='sgd',
                      metrics=['accuracy'])
        print(model.summary())
        return model
    return compiler

@compile_classifier
def make_Loc(input_shape=(64,64, 1)):
    ''' Return Keras model analogous to Loc architecture [see reference above].
    '''
    model = Sequential()
    model.add(MaxPooling2D((1,1), input_shape=input_shape))
    add_convblock(model, depth=2, channels= 32, kernel=5, pool=2) # now  32 x 32 x 32
    add_convblock(model, depth=2, channels= 32, kernel=5, pool=4) # now   8 x  8 x 64
    add_loclblock(model, depth=2, channels=128, kernel=3, pool=1) # now   4 x  4 x128
    add_dense_classifier(model, hidden_sizes=[1024, 1024])
    return model

@compile_classifier       
def make_VGG16(input_shape=(64,64, 1)):
    ''' Return Keras model analogous to VGG16.
    '''
    model = Sequential()
    model.add(MaxPooling2D((1,1), input_shape=input_shape))
    add_convblock(model, depth=2, channels= 32, kernel=5, pool=2) # now  32 x 32 x 32
    add_convblock(model, depth=2, channels= 64, kernel=5, pool=2) # now  16 x 16 x 64
    add_convblock(model, depth=2, channels=128, kernel=5, pool=4) # now   4 x  4 x128
    add_dense_classifier(model, hidden_sizes=[1024, 1024])
    return model

@compile_classifier       
def make_MLP(input_shape=(64,64, 1)):
    ''' Return multi-layer perceptron.
    '''
    model = Sequential()
    model.add(MaxPooling2D((1,1), input_shape=input_shape))
    add_dense_classifier(model, hidden_sizes=[1024, 1024])
    return model
