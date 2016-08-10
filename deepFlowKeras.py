import pandas as pd
import numpy as np
import sys
from keras.models import Sequential
from keras.layers import Dense, Merge, Activation, Convolution2D, Flatten, Reshape

FCDIM = 256
FTSIZE = 4
BUFLEN = 16
numft = 4

epoch = 20
batch = 32

features = ['cip','sip','cp','sp','cb','sb']
features_comp = ['cp','sp']
X_train = []

loc = 'binary/'
loc_comp = 'binary/compressed/'
part = '_p0'

def load_data():
	global loc, loc_comp, part
	global X_train, Y_train
	global features, features_compressed
	for feature in features:
		if feature in features_comp:
			X_train.append(np.load(loc_comp+feature+part+'.npy'))
		else:
			X_train.append(np.load(loc+feature+part+'.npy'))
		print 'X train ('+feature+'): ',
		print X_train[len(X_train) - 1].shape
	Y_train = np.load(loc+'tr_2kb'+part+'.npy')
	print 'Y train: ',
	print Y_train.shape

def create_model():
	global X_train
	"""
	layer_sb = Sequential([
			Convolution2D(numft, FTSIZE, 257, border_mode='same', input_shape=(1, BUFLEN, 257)),
			Activation('relu'),
			Flatten(),
			Dense(FCDIM)
	])
	layer_cb = Sequential([
			Convolution2D(numft, FTSIZE, 257, border_mode='same', input_shape=(1, BUFLEN, 257)),
			Activation('relu'),
			Flatten(),
			Dense(FCDIM)
	])
	"""
	layer_sb = Sequential([
			Reshape((BUFLEN*257,), input_shape=(1, BUFLEN, 257)),
			Dense(FCDIM),
	])
	layer_cb = Sequential([
			Reshape((BUFLEN*257,), input_shape=(1, BUFLEN, 257)),
			Dense(FCDIM),
	])

	layer_cip = Sequential([Dense(FCDIM, input_dim=(X_train[features.index('cip')].shape[1]))])
	layer_sip = Sequential([Dense(FCDIM, input_dim=(X_train[features.index('sip')].shape[1]))])
	layer_cp = Sequential([Dense(FCDIM, input_dim=(X_train[features.index('cp')].shape[1]))])
	layer_sp = Sequential([Dense(FCDIM, input_dim=(X_train[features.index('sp')].shape[1]))])

	layer_merged = Merge([layer_cip,layer_sip,layer_cp,layer_sp,layer_cb,layer_sb],
			mode = 'sum')

	total_model = Sequential()
	total_model.add(layer_merged)
	total_model.add(Activation('relu'))
	total_model.add(Dense(FCDIM))
	total_model.add(Activation('relu'))
	total_model.add(Dense(2))
	total_model.add(Activation('softmax'))

	return total_model

print 'Reading binary inputs...'
load_data()

print 'Compiling Neural Network Model...'
model = create_model()
model.compile(loss='binary_crossentropy', optimizer='sgd', metrics=['accuracy'])

print 'Training Neural Network Model...'
model.fit(X_train,Y_train, batch_size=batch, nb_epoch=epoch)
# model.evaluate(X_train,Y_train, batch_size=batch)
Y_predict = model.predict(X_train)
TP, FP, TN, FN = 0, 0, 0, 0
for i in range(len(Y_predict)):
	predict = Y_predict[i][0] < Y_predict[i][1]
	truth = Y_train[i][0] < Y_train[i][1]
	if predict and truth:
		TN += 1
	else:
		if predict:
			FN += 1
		elif truth:
			FP += 1
		else:
			TP += 1

print '[TP FP TN FN] = ['+str(TP)+' '+str(FP)+' '+str(TN)+' '+str(FN)+']'

