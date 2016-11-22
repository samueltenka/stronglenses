''' author: sam tenka, daniel zhang
    credits: Thanks to the following:
        Convolutional Layers in Keras: https://keras.io/layers/convolutional/
        VGG16 archictecture: https://gist.github.com/baraldilorenzo/07d7802847aaad0a35d3
        Loc architecture: http://cs231n.stanford.edu/reports/cpd_final.pdf
    date: 2016-10-12
    descr: Image-classification architectures for Strong Lenses.
    usage: Import model-constructors as follows: 
        from model.make_model import make_mlp
        from model.make_model import make_mlp_wide
        from model.make_model import make_shallow_res
'''

from __future__ import print_function

from keras.models import Sequential, Model
from keras.layers import Input, Flatten, Dense, Activation, Dropout, merge
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.layers.local import LocallyConnected2D

def compile_classifier(make_model):
    ''' Decorate `make_model` to compile and summarize model.

        We use adadelta with Keras defaults.
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
def make_mlp(input_shape=(64,64,3)):
    ''' Multilayer perceptron with two hidden layers.
    '''
    x = Input(shape=input_shape) 
    f = Flatten()(x)

    d0 = Dropout(0.5)(f)
    h0 = Dense(64, activation='softplus')(d0)
    d1 = Dropout(0.5)(h0)
    h1 = Dense(64, activation='softplus')(d1)
    y = Dense(1, activation='sigmoid')(h1)
    return Model(input=x, output=y)

@compile_classifier 
def make_mlp_wide(input_shape=(64,64,3)):
    ''' Multilayer perceptron with two hidden layers.
    '''
    x = Input(shape=input_shape) 
    f = Flatten()(x)

    d0 = Dropout(0.5)(f)
    h0 = Dense(256, activation='softplus')(d0)
    d1 = Dropout(0.5)(h0)
    h1 = Dense(64, activation='softplus')(d1)
    y = Dense(1, activation='sigmoid')(h1)
    return Model(input=x, output=y)

@compile_classifier 
def make_shallow_res(input_shape=(64,64,3)):
    ''' Return model with one resnet block followed by two dense layers.
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


