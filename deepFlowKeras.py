import pandas as pd
import numpy as np
import sys
from keras.models import Sequential
from keras.layers import Dense, Merge, Activation, Convolution2D, Flatten

FCDIM = 128
FTSIZE = 4
BUFLEN = 16
numft = 4
epoch = 4

features = ['cip','sip','cp','sp','cb','sb']
X_train = {}

def create_model():
	layer_sb = Sequential([
			Convolution2D(numft, FTSIZE, 256, border_mode='same', input_shape=(1, BUFLEN, 256)),
			Activation('relu'),
			Flatten(),
			Dense(FCDIM)
	])
	layer_cb = Sequential([
			Convolution2D(numft, FTSIZE, 256, border_mode='same', input_shape=(1, BUFLEN, 256)),
			Activation('relu'),
			Flatten(),
			Dense(FCDIM)
	])

	layer_cip = Sequential([Dense(FCDIM, input_dim=(4*256))])
	layer_sip = Sequential([Dense(FCDIM, input_dim=(4*256))])
	layer_cp = Sequential([Dense(FCDIM, input_dim=1024)])
	layer_sp = Sequential([Dense(FCDIM, input_dim=1024)])

	layer_merged = Merge([layer_cip,layer_sip,layer_cp,layer_sp,layer_cb,layer_sb],
			mode = 'sum')

	total_model = Sequential()
	total_model.add(layer_merged)
	total_model.add(Activation('relu'))
	total_model.add(Dense(2))
	total_model.add(Activation('relu'))

	return total_model

print 'Reading binary inputs...'
for feature in features:
	feature = feature + '_p0'
	X_train[feature] = np.load('binary/'+feature+'.npy')
	print 'X train ('+feature+'): ',
	print X_train[feature].shape

#print 'Compiling Neural Network Model...'
#model = create_model()
#model.compile(loss='binary_crossentropy', optimizer='sgd', metrics=['accuracy'])

#print 'Training Neural Network Model...'
#total_model.fit(X,Y)
#total_model.evaluate(X,Y)
#total_model.predict(X)
