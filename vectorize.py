import pandas as pd
import numpy as np
import sys

loc = 'binary'
part = ''

threshold = 16 # KB

def vector(value,size):
	v = [0]*size
	v[value] = 1
	return v

def ip_vectorize(df_col):
	v = map(lambda x: map(lambda y: vector(int(y), 256), x.split('.')), df_col.values)
	print np.array(v).shape
	return np.array(v).reshape((len(v), 256*4))

def port_vectorize(df_col):
	v = map(lambda x: vector(x+1, 1025) if (x < 1024) else vector(0, 1025), df_col.values)
	print np.array(v).shape
	return np.array(v)

def buf_vectorize(df_col):
	v = map(lambda x: map(lambda y: vector(y+1, 257), x), df_col.values)
	print np.array(v).shape
	return np.array(v).reshape((len(v), 1, 16, 257))

def truth_vectorize(df_col):
	global threshold
	v = map(lambda x: vector(x, 2), ((df_col >= (threshold*1024)) + 0).values)
	print np.array(v).shape
	return np.array(v)

print 'Reading flow data...'
df = pd.read_csv('flowData/flow_dump.csv')
df_cb = pd.read_csv('flowData/flow_dump_cb.csv').drop('dummy',axis=1)[0:100000]
df_sb = pd.read_csv('flowData/flow_dump_sb.csv').drop('dummy',axis=1)[0:100000]
print 'Vectorizing data...'

#np.save(loc+'/sip'+part,ip_vectorize(df.sip))
#np.save(loc+'/cip'+part,ip_vectorize(df.cip))
#np.save(loc+'/cp'+part,port_vectorize(df.cp))
#np.save(loc+'/sp'+part,port_vectorize(df.sp))
#np.save(loc+'/cb'+part,buf_vectorize(df_cb))
#np.save(loc+'/sb'+part,buf_vectorize(df_sb))
np.save(loc+'/tr_'+str(threshold)+'kb'+part,truth_vectorize(df.flow_size))
print 'done'

#total_model.fit(X,Y)
#total_model.evaluate(X,Y)
#total_model.predict(X)
