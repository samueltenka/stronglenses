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
        from model.model import make_Loc
        from model.model import make_VGG16
        from model.model import make_MLP
'''

from __future__ import print_function

from keras.models import Sequential, Model
from keras.layers import Input, Flatten, Dense, Activation, Dropout, merge
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.layers.local import LocallyConnected2D

def compile_classifier(make_model):
    ''' Decorate `make_model` to compile and summarize model.
    '''
    def compiler(*args, **kwargs):
        model = make_model(*args, **kwargs)
        model.compile(loss='binary_crossentropy',
                      optimizer='adadelta',
                      metrics=['accuracy'])
        print(model.summary())
        return model
    return compiler

@compile_classifier       
def make_MLP(input_shape=(64,64,3)):
    ''' Return multi-layer perceptron.
    '''
    x = Input(shape=input_shape) 
    c0 = Convolution2D(16, 5, 5, border_mode='valid')(x)
    b0 = BatchNormalization()(c0)
    a0 = Activation('softplus')(b0)
    m0 = MaxPooling2D((3, 3))(a0)
    f = Flatten()(m0)
    d0 = Dropout(0.5)(f)
    h0 = Dense(256, activation='softplus')(d0)
    h1 = Dense(64, activation='softplus')(h0)
    y = Dense(1, activation='sigmoid')(h1)
    return Model(input=x, output=y) 
