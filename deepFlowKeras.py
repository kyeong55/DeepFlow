import pandas as pd
import numpy as np
import sys
from keras import backend as K
from keras.models import Sequential
from keras.layers import Dense, Merge, Activation, Convolution2D, Flatten, Reshape
import tensorflow as tf
import metricPreRec as m
from sklearn.metrics import (precision_score, recall_score)

threshold = 8 # KB

FCDIM = 128
FCNUM = 2

FTSIZE = 4
BUFLEN = 16
numft = 4

epoch = 100
test_epoch = 1
batch = 32
ver = 0

loc = 'binary/http/'
log_file_name = 'log_'+str(epoch)+'_'+str(threshold)+'k_'+sys.argv[1]
log_file = 'log/' + log_file_name + '.csv'
dump_file = 'log/dump/' + log_file_name
log = open(log_file,'w')
log.write('Epoch,Precision_S,Recall_S,Precision_L,Recall_L\n')
log.close()

features = ['cip','sip','sp','cb','sb']
X_train, Y_train = [], []
X_test, Y_test = [], []
X = []
Y = np.empty(0)

input_shape = {}

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
#	layer_cp = Sequential([Dense(FCDIM, input_dim=(input_shape['cp'][0]))])
	layer_sp = Sequential([Dense(FCDIM, input_dim=(input_shape['sp'][0]))])

#	layer_merged = Merge([layer_cip,layer_sip,layer_cp,layer_sp,layer_cb,layer_sb],
	layer_merged = Merge([layer_cip,layer_sip,layer_sp,layer_cb,layer_sb],
			mode = 'sum')

	total_model = Sequential()
	total_model.add(layer_merged)
	total_model.add(Activation('relu'))
	for i in range(FCNUM - 1):
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
	return Y_predict

def run(train, test):
	global input_shape
	global epoch, batch
	global TP,FP,TN,FN
	global log

	print '=== Train Set: ',
	print train

	# Compile
	model = create_model()
	model.compile(loss='binary_crossentropy', optimizer='sgd', metrics=['accuracy'])
#	model.compile(loss='binary_crossentropy', optimizer='sgd', metrics=m.get_metrics())

	TP, FP, TN, FN = 0.0, 0.0, 0.0, 0.0

	# Dump test results continuously
	results = []

	# Train
	for e in range(epoch):
		print '=== Epoch ' + str(e+1) + '/' + str(epoch)
		for i in train:
			load_data(i)
			# print 'Train (' + str(i) + '/' + str(len(train)) + ')'
			model.fit(X, Y, batch_size=batch, nb_epoch=1, verbose=ver)
		if ((e+1)%test_epoch == 0):
			predict = []
			TP, FP, TN, FN = 0.0, 0.0, 0.0, 0.0
			for i in test:
				load_data(i)
				predict.append(test_model(model, X, Y))
			results.append(np.concatenate(predict, axis=0))
			log = open(log_file,'a')
			printLog = str(e+1)+','+str(TP/(TP+FP))+','+str(TP/(TP+FN))+','+str(TN/(TN+FN))+','+str(TN/(TN+FP))+'\n'
			log.write(printLog)
			log.close()

	np.save(dump_file, np.array(results))

	# Test
	"""
	for i in test:
		load_data(i)
		# print 'Test (' + str(i) + '/' + str(len(test)) + ')'
		test_model(model, X, Y)
	"""

	# results
	print '[TP FP TN FN] = ['+str(int(TP))+' '+str(int(FP))+' '+str(int(TN))+' '+str(int(FN))+']'
	print 'Precision(S):\t' + str(TP/(TP+FP))
	print 'Recall(S):\t' + str(TP/(TP+FN))
	print 'Precision(L):\t' + str(TN/(TN+FN))
	print 'Recall(L):\t' + str(TN/(TN+FP))

def addResultInfo():
	global log_file_name
	info = open('log/info','a')
	info.write(log_file_name)
	info.write('\n\t- FCDIM: '+str(FCDIM))
	info.write('\n\t- FClayer: '+str(FCNUM))
	info.write('\n\t- DATA: '+str(loc[7:])+'\n')
	info.close()

print '- Threshold: ' + str(threshold) + 'KB'
print '- epoch: ' + str(epoch),
print ', batch: ' + str(batch)
get_input_shape()

run(range(5),range(5))

addResultInfo()
print 'Finish'

"""
for i in range(5):
	train = range(i, (i+1))
	test = range(0, i) + range((i+1), 5)
	run(train,test)
"""
# model.evaluate(X_train,Y_train, batch_size=batch)
