''' author: daniel zhang and samuel tenka
    date: 2016-12-01
    descr: Test Keras by building a toy network.
           See stronglenses/model/make_model.py for
           our actual models.
'''

#original script for simpleConvNN also known as jeremy.
from keras.models import Sequential 
from keras.layers import Flatten, Dense, Activation
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.utils.np_utils import to_categorical
import numpy as np
from sklearn.metrics import roc_auc_score

#'Jeremy'
def simpleConvNN(input_shape=(64, 64, 3)):
	model = Sequential()
	model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), input_shape=input_shape))
	#model went from 64x64x3 to 32x32x3
	model.add(Convolution2D(64, 3, 3, subsample=(2,2), activation='softplus'))
	#model is now 16x16x64
	model.add(Convolution2D(32, 3, 3, activation='softplus'))
	#model is now 16x16x32
	model.add(Convolution2D(16, 3, 3, activation='softplus'))
	#model is now 16x16x16
	model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))
	#model is now  8x8x16
	model.add(Flatten())
	#model is now 1024 (flattened from 8x8x16)
	model.add(Dense(256, activation='softplus'))
	model.add(Dense(32, activation='softplus'))
	model.add(Dense(2, activation='softmax'))
	return model

optimizer = 'adam'

#make model and compile
model = simpleConvNN()
#model.compile(optimizer='sgd',
model.compile(optimizer=optimizer,
	loss='categorical_crossentropy',
	metrics=['accuracy'])

X = np.load('Large_Data_Set/xTrain_full.npy')
Y = np.load('Large_Data_Set/yTrain_full.npy')
testX = np.load('Large_Data_Set/xTest_full.npy')
testY = np.load('Large_Data_Set/yTest_full.npy')

Y = to_categorical(Y)
testY = to_categorical(testY)

print '\nUsing', optimizer, 'optimizer'
print model.summary()
print '\n'
model.fit(X, Y, validation_data=(testX, testY))

#AUROC after training
y_prob = model.predict_proba(testX) #900 x 2
y_prob = y_prob.transpose() #2 x 900
print '\nAUROC:', roc_auc_score(testY.transpose()[1], y_prob[1])

