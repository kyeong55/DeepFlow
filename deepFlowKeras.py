import pandas as pd
import numpy as np
import sys
from keras.models import Sequential
from keras.layers import Dense, Merge, Activation, Convolution2D, Flatten, Reshape

threshold = 8 # KB

FCDIM = 256
FTSIZE = 4
BUFLEN = 16
numft = 4

epoch = 4
batch = 32

loc = 'binary/nothttp/'

features = ['cip','sip','cp','sp','cb','sb']
X_train, Y_train = [], []
X_test, Y_test = [], []
X = []
Y = np.empty(0)

input_shape = {}

def load_data_all():
	global loc
	global X_train, Y_train
	global features

	for part in train:
		part = '_' + str(part)
		X = []
		for feature in features:
			X.append(np.load(loc+feature+part+'.npy'))
			if part == '_0':
				input_shape[feature] = X[len(X)-1].shape[1:]
				print 'X ('+feature+'):',
				print input_shape[feature]
		X_train.append(X)
		Y_train.append(np.load(loc+'tr_'+str(threshold)+'kb'+part+'.npy'))

	for part in test:
		part = '_' + str(part)
		X = []
		for feature in features:
			X.append(np.load(loc+feature+part+'.npy'))
		X_test.append(X)
		Y_test.append(np.load(loc+'tr_'+str(threshold)+'kb'+part+'.npy'))

def load_data(num):
	global loc
	global X,Y
	global features
	part = '_' + str(num)
	X = []
	for feature in features:
		X.append(np.load(loc+feature+part+'.npy'))
	Y = np.load(loc+'tr_'+str(threshold)+'kb'+part+'.npy')

def get_input_shape():
	global X
	global features
	global input_shape
	load_data(0)
	i = 0
	print '- Input shape:'
	for feature in features:
		input_shape[feature] = X[i].shape[1:]
		print '  ('+feature+'):',
		print input_shape[feature]
		i += 1

def create_model():
	global FCDIM, FTSIZE, BUFLEN, numft
	global input_shape
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
			Dense(FCDIM)
	])
	layer_cb = Sequential([
			Reshape((BUFLEN*257,), input_shape=(1, BUFLEN, 257)),
			Dense(FCDIM)
	])

	layer_cip = Sequential([Dense(FCDIM, input_dim=(input_shape['cip'][0]))])
	layer_sip = Sequential([Dense(FCDIM, input_dim=(input_shape['sip'][0]))])
	layer_cp = Sequential([Dense(FCDIM, input_dim=(input_shape['cp'][0]))])
	layer_sp = Sequential([Dense(FCDIM, input_dim=(input_shape['sp'][0]))])

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

TP, FP, TN, FN = 0.0, 0.0, 0.0, 0.0

def test_model(model, X_test, Y_test):
	global TP,FP,TN,FN
	Y_predict = model.predict(X_test)
	for i in range(len(Y_predict)):
		predict = Y_predict[i][0] < Y_predict[i][1]
		truth = Y_test[i][0] < Y_test[i][1]
		if predict and truth:
			TN += 1
		else:
			if predict:
				FN += 1
			elif truth:
				FP += 1
			else:
				TP += 1

def run(train, test):
	global input_shape
	global epoch, batch
	global TP,FP,TN,FN

	print '=== Train Set: ',
	print train

	print 'Compiling Neural Network Model...'
	model = create_model()
	model.compile(loss='binary_crossentropy', optimizer='sgd', metrics=['accuracy'])

	TP, FP, TN, FN = 0.0, 0.0, 0.0, 0.0

	sys.stdout.write("\033[F")
	print 'Training Neural Network Model...'
	for e in range(epoch):
		for i in train:
			load_data(i)
			# print 'Train (' + str(i) + '/' + str(len(train)) + ')'
			model.fit(X, Y, batch_size=batch, nb_epoch=1, verbose=0)

	sys.stdout.write("\033[F")
	print 'Testing Trained Model...'
	for i in test:
		load_data(i)
		# print 'Test (' + str(i) + '/' + str(len(test)) + ')'
		test_model(model, X, Y)

	# results
	sys.stdout.write("\033[F")
	print '[TP FP TN FN] = ['+str(int(TP))+' '+str(int(FP))+' '+str(int(TN))+' '+str(int(FN))+']'
	print 'Precision(S):\t' + str(TP/(TP+FP))
	print 'Recall(S):\t' + str(TP/(TP+FN))
	print 'Precision(L):\t' + str(TN/(TN+FN))
	print 'Recall(L):\t' + str(TN/(TN+FP))

print '- Threshold: ' + str(threshold) + 'KB'
print '- epoch: ' + str(epoch),
print ', batch: ' + str(batch)
get_input_shape()

#run(range(20),range(20))

"""
for i in range(5):
	train = range(i*4, (i+1)*4)
	test = range(0, i*4) + range((i+1)*4, 20)
	run(train,test)
"""
"""
for i in range(3):
	train = range(i*3, (i+1)*3)
	test = range(0, i*3) + range((i+1)*3, 9)
	run(train,test)
"""
run(range(9),range(9))

# model.evaluate(X_train,Y_train, batch_size=batch)
