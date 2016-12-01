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
from keras.layers.convolutional import Convolution2D, MaxPooling2D, AveragePooling2D, ZeroPadding2D
from keras.layers.local import LocallyConnected2D
from keras.regularizers import l2
import numpy as np

def compile_classifier(optimizer = 'adadelta'):
    ''' Decorate `make_model` to compile and summarize model.

        We use adadelta with Keras' default parameters.
    '''
    def model_decorator(make_model):
        def compiler(*args, **kwargs):
            model = make_model(*args, **kwargs)
            model.compile(loss='binary_crossentropy',
                          optimizer=optimizer,
                          metrics=['accuracy'])
            print(model.summary())
            return model
        return compiler
    return model_decorator

@compile_classifier() 
def make_logistic(input_shape=(64,64,3)):
    ''' Return logistic regressor.
    '''
    x = Input(shape=input_shape) 
    f = Flatten()(x)
    y = Dense(1, W_regularizer=l2(1.0), activation='sigmoid')(f)
    return Model(input=x, output=y)

@compile_classifier() 
def make_mlp_l2(input_shape=(64,64,3)):
    ''' Multilayer perceptron that takes a softened max
        (not a softmax) of 4 L2-regularized perceptrons.
    '''
    x = Input(shape=input_shape) 
    f = Flatten()(x)
    hs = [Dense(1, W_regularizer=l2(0.5), activation='sigmoid')(f) for i in range(4)]
    y = merge(hs, mode=lambda x: ((x[0]**4+x[1]**4+x[2]**4+x[3]**4)/4)**0.25, output_shape=(1,))
    return Model(input=x, output=y)

@compile_classifier() 
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


@compile_classifier() 
def make_mlp_wide(input_shape=(64,64,3)):
    ''' Multilayer perceptron with two hidden layers,
        increased size of initial layer.
    '''
    x = Input(shape=input_shape) 
    f = Flatten()(x)

    d0 = Dropout(0.5)(f)
    h0 = Dense(256, activation='softplus')(d0)
    d1 = Dropout(0.5)(h0)
    h1 = Dense(64, activation='softplus')(d1)
    y = Dense(1, activation='sigmoid')(h1)
    return Model(input=x, output=y)

@compile_classifier() 
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

def make_res(dim, ker, poolker, depth=2, activation='relu'):
    ''' Return a tensor-transformer representing a residual block. '''
    def res(in_layer):
        a = in_layer
        for d in range(depth):
            c = Convolution2D(dim, ker, ker, border_mode='same', init='zero')(a)
            b = BatchNormalization()(c)
            a = Activation(activation)(b)
        s = merge([in_layer, a], mode='sum')
        m = MaxPooling2D((poolker, poolker))(s)
        return m
    return res

def make_dense(dims, activation):
    ''' Return a tensor-transformer representing a dense block. '''
    def dense(in_layer):
        o = in_layer
        for dim in dims:
            d = Dropout(0.5)(o) 
            o = Dense(dim, activation=activation)(d)
        return o
    return dense

@compile_classifier() 
def make_res_2(input_shape=(64,64,3)):
    ''' Return model with two resnet blocks followed by two dense layers.
    '''
    x = Input(shape=input_shape) 
    c = Convolution2D(16, 3, 3, border_mode='same', activation='relu')(x)
    m = MaxPooling2D((3, 3))(c)

    r0 = make_res(dim=16, ker=2, depth=2, poolker=2)(m) 
    r1 = make_res(dim=16, ker=2, depth=2, poolker=2)(r0)

    f = Flatten()(r1)
    y = make_dense(dims=[64, 64, 1], activation='sigmoid')(f)

    return Model(input=x, output=y)

@compile_classifier() 
def make_squeeze_res_old(input_shape=(64,64,3)):
    ''' Basic convolutional model with aggressive, ealy subsampling.
    '''
    x = Input(shape=input_shape) 
    c0 = Convolution2D(64, 4, 4, subsample=(4, 4), activation='softplus', border_mode='same')(x)
    c1 = Convolution2D(32, 3, 3, subsample=(2, 2), activation='softplus', border_mode='same')(c0)
    c2 = Convolution2D(16, 3, 3, subsample=(2, 2), activation='softplus', border_mode='same')(c1)
    f = Flatten()(c2)
    y = make_dense(dims=[256, 32, 1], activation='sigmoid')(f)
    return Model(input=x, output=y)

@compile_classifier() 
def make_squeeze_res(input_shape=(64,64,3)):
    ''' Return model with one resnet block followed by two dense layers.
    '''
    x = Input(shape=input_shape) 
    c = Convolution2D(8, 4, 4, subsample=(4, 4), border_mode='same')(x)

    c1 = Convolution2D(8, 3, 3, border_mode='same')(c)
    b1 = BatchNormalization()(c1)
    a1 = Activation('softplus')(b1)
    c2 = Convolution2D(8, 3, 3, border_mode='same')(a1)
    b2 = BatchNormalization()(c2)
    a2 = Activation('softplus')(b2)
    s0 = merge([c, a2], mode='sum')
    m1 = MaxPooling2D((3, 3))(s0)

    f = Flatten()(c2)
    y = make_dense(dims=[256, 32, 1], activation='sigmoid')(f)
    return Model(input=x, output=y)

@compile_classifier() 
def make_squeeze_res_wide(input_shape=(64,64,3)):
    ''' Return model with one resnet block followed by two dense layers.
    '''
    x = Input(shape=input_shape) 
    c0 = Convolution2D(32, 3, 3, activation='softplus', subsample=(2, 2), border_mode='same')(x)
    b0 = BatchNormalization()(c0)
    c1 = Convolution2D(16, 3, 3, activation='softplus', subsample=(2, 2), border_mode='same')(b0)
    b1 = BatchNormalization()(c1)
    c2 = Convolution2D( 8, 3, 3, activation='softplus', subsample=(2, 2), border_mode='same')(b1)

    m = MaxPooling2D((4, 4))(b0)
    s = merge([c2, m], mode='concat')
    f = Flatten()(s)

    d0 = Dropout(0.5)(f)
    z0 = Dense(128, activation='softplus')(d0)
    z1 = Dense(32, activation='softplus')(z0)
    y = Dense(1, activation='sigmoid')(z1)
    return Model(input=x, output=y)

@compile_classifier() 
def make_squeeze_skip(input_shape=(64,64,3)):
    ''' Return model with one resnet block followed by two dense layers.
    '''
    x = Input(shape=input_shape) 
    c0 = Convolution2D(32, 3, 3, activation='softplus', subsample=(4, 4), border_mode='same')(x)
    b0 = BatchNormalization()(c0)
    d0 = Dropout(0.5)(b0)
    c1 = Convolution2D( 8, 3, 3, activation='softplus', subsample=(2, 2), border_mode='same')(d0)
    b1 = BatchNormalization()(c1)
    d1 = Dropout(0.5)(b1)

    f = Flatten()(d1)
    z0 = Dense(128, activation='softplus')(f)
    z1 = Dense(32, activation='softplus')(z0)
    y = Dense(1, activation='sigmoid')(z1)
    return Model(input=x, output=y)

@compile_classifier('adam')
def make_softplus_3(input_shape=(64, 64, 3)):
    ''' Basic convolutional model with 3 convolutional layers.
        Trained using 'adam'.
    '''
    model = Sequential() # 64x64x3
    model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), input_shape=input_shape)) # 32x32x3
    model.add(Convolution2D(64, 3, 3, subsample=(2,2), activation='softplus')) # 16x16x64
    model.add(Convolution2D(32, 3, 3, activation='softplus')) # 16x16x32
    model.add(Convolution2D(16, 3, 3, activation='softplus')) # 16x16x16
    model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2))) # 8x8x16
    model.add(Flatten()) # 1024
    model.add(Dense(256, activation='softplus'))
    model.add(Dense(32, activation='softplus'))
    model.add(Dense(1, activation='sigmoid'))
    return model
    
