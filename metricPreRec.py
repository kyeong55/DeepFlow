import numpy as np
from keras import backend as K

def pre_S(y_true, y_pred):
	s_flow = K.variable(np.array([1,0]))
	p = K.cast(K.equal(K.argmax(s_flow, axis=-1), K.argmax(y_pred, axis=-1)), K.floatx())
	n = K.cast(K.not_equal(K.argmax(s_flow, axis=-1), K.argmax(y_pred, axis=-1)), K.floatx())
	t = K.cast(K.equal(K.argmax(y_true, axis=-1), K.argmax(y_pred, axis=-1)), K.floatx())
	f = K.cast(K.not_equal(K.argmax(y_true, axis=-1), K.argmax(y_pred, axis=-1)), K.floatx())
	tp = t*p
	fp = f*p
	return K.sum(tp) / (K.sum(tp) + K.sum(fp))
	
def rec_S(y_true, y_pred):
	s_flow = K.variable(np.array([1,0]))
	p = K.cast(K.equal(K.argmax(s_flow, axis=-1), K.argmax(y_pred, axis=-1)), K.floatx())
	n = K.cast(K.not_equal(K.argmax(s_flow, axis=-1), K.argmax(y_pred, axis=-1)), K.floatx())
	t = K.cast(K.equal(K.argmax(y_true, axis=-1), K.argmax(y_pred, axis=-1)), K.floatx())
	f = K.cast(K.not_equal(K.argmax(y_true, axis=-1), K.argmax(y_pred, axis=-1)), K.floatx())
	tp = t*p
	fn = f*n
	return K.sum(tp) / (K.sum(tp) + K.sum(fn))

def pre_L(y_true, y_pred):
	s_flow = K.variable(np.array([1,0]))
	p = K.cast(K.equal(K.argmax(s_flow, axis=-1), K.argmax(y_pred, axis=-1)), K.floatx())
	n = K.cast(K.not_equal(K.argmax(s_flow, axis=-1), K.argmax(y_pred, axis=-1)), K.floatx())
	t = K.cast(K.equal(K.argmax(y_true, axis=-1), K.argmax(y_pred, axis=-1)), K.floatx())
	f = K.cast(K.not_equal(K.argmax(y_true, axis=-1), K.argmax(y_pred, axis=-1)), K.floatx())
	tn = t*n
	fn = f*n
	return K.sum(tn) / (K.sum(tn) + K.sum(fn))
	
def rec_L(y_true, y_pred):
	s_flow = K.variable(np.array([1,0]))
	p = K.cast(K.equal(K.argmax(s_flow, axis=-1), K.argmax(y_pred, axis=-1)), K.floatx())
	n = K.cast(K.not_equal(K.argmax(s_flow, axis=-1), K.argmax(y_pred, axis=-1)), K.floatx())
	t = K.cast(K.equal(K.argmax(y_true, axis=-1), K.argmax(y_pred, axis=-1)), K.floatx())
	f = K.cast(K.not_equal(K.argmax(y_true, axis=-1), K.argmax(y_pred, axis=-1)), K.floatx())
	tn = t*n
	fp = f*p
	return K.sum(tn) / (K.sum(tn) + K.sum(fp))

def tp(y_true, y_pred):
	s_flow = K.variable(np.array([1,0]))
	p = K.cast(K.equal(K.argmax(s_flow, axis=-1), K.argmax(y_pred, axis=-1)), K.floatx())
	t = K.cast(K.equal(K.argmax(y_true, axis=-1), K.argmax(y_pred, axis=-1)), K.floatx())
	tr_p = t*p
	return K.sum(tr_p)

def tn(y_true, y_pred):
	s_flow = K.variable(np.array([1,0]))
	n = K.cast(K.not_equal(K.argmax(s_flow, axis=-1), K.argmax(y_pred, axis=-1)), K.floatx())
	t = K.cast(K.equal(K.argmax(y_true, axis=-1), K.argmax(y_pred, axis=-1)), K.floatx())
	tr_n = t*n
	return K.sum(tr_n)

def fp(y_true, y_pred):
	s_flow = K.variable(np.array([1,0]))
	p = K.cast(K.equal(K.argmax(s_flow, axis=-1), K.argmax(y_pred, axis=-1)), K.floatx())
	f = K.cast(K.not_equal(K.argmax(y_true, axis=-1), K.argmax(y_pred, axis=-1)), K.floatx())
	fl_p = f*p
	return K.sum(fl_p)

def fn(y_true, y_pred):
	s_flow = K.variable(np.array([1,0]))
	n = K.cast(K.not_equal(K.argmax(s_flow, axis=-1), K.argmax(y_pred, axis=-1)), K.floatx())
	f = K.cast(K.not_equal(K.argmax(y_true, axis=-1), K.argmax(y_pred, axis=-1)), K.floatx())
	fl_n = f*n
	return K.sum(fl_n)

def acc(y_true, y_pred):
	return K.mean(K.equal(K.argmax(y_true, axis=-1), K.argmax(y_pred, axis=-1)))

def get_metrics():
#	return ['accuracy']
#	return [pre_S, rec_S, pre_L, rec_L]
#	return [tp, fp, tn, fn]
	return [acc]
