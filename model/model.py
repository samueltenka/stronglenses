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
    c0 = Convolution2D(8, 3, 3, border_mode='same')(x)
    m0 = MaxPooling2D((3, 3))(c0)

    c1 = Convolution2D(8, 3, 3, border_mode='same')(m0)
    b1 = BatchNormalization()(c1)
    a1 = Activation('relu')(b1)
    c2 = Convolution2D(8, 3, 3, border_mode='same')(a1)
    b2 = BatchNormalization()(c2)
    a2 = Activation('relu')(b2)
    s0 = merge([m0, a2], mode='sum')
    m1 = MaxPooling2D((3, 3))(s0)

    f = Flatten()(m1)

    d0 = Dropout(0.5)(f)
    h0 = Dense(64, activation='sigmoid')(d0)
    d1 = Dropout(0.5)(h0)
    h1 = Dense(64, activation='sigmoid')(d1)
    y = Dense(1, activation='sigmoid')(h1)
    return Model(input=x, output=y)






@compile_classifier
def make_res(input_shape=(64,64,3)):
    ''' Return multi-layer perceptron.
    '''
    x = Input(shape=input_shape) 

    c0 = Convolution2D(32, 5, 5, border_mode='same')(x)
    b0 = BatchNormalization()(c0)
    a0 = Activation('relu')(b0)
    c1 = Convolution2D(32, 5, 5, border_mode='same')(a0)
    b1 = BatchNormalization()(c1)
    a1 = Activation('relu')(b1)
    s0 = merge([c0, a1], mode='sum')
    m0 = MaxPooling2D((3, 3))(s0)

    c2 = Convolution2D(16, 5, 5, border_mode='same')(m0)
    b2 = BatchNormalization()(c2)
    a2 = Activation('relu')(b2)
    c3 = Convolution2D(16, 5, 5, border_mode='same')(a2)
    b3 = BatchNormalization()(c3)
    a3 = Activation('relu')(b3)
    s1 = merge([c2, a3], mode='sum')
    m1 = MaxPooling2D((3, 3))(s1)

    f = Flatten()(m1)

    d0 = Dropout(0.5)(f)
    h0 = Dense(64, activation='sigmoid')(d0)
    d1 = Dropout(0.5)(h0)
    h1 = Dense(64, activation='sigmoid')(d1)
    y = Dense(1, activation='sigmoid')(h1)
    return Model(input=x, output=y) 
