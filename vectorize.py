import pandas as pd
import numpy as np
import sys


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

print 'Reading flow data...'
df = pd.read_csv('flowData/flow_dump.csv')[0:10000]
df_cb = pd.read_csv('flowData/flow_dump_cb.csv').drop('dummy',axis=1)[0:10000]
df_sb = pd.read_csv('flowData/flow_dump_sb.csv').drop('dummy',axis=1)[0:10000]
print 'Vectorizing data...'

loc = 'binary'
part = '_p0'
np.save(loc+'/sip'+part,ip_vectorize(df.sip))
np.save(loc+'/cip'+part,ip_vectorize(df.cip))
np.save(loc+'/cp'+part,port_vectorize(df.cp))
np.save(loc+'/sp'+part,port_vectorize(df.sp))
np.save(loc+'/cb'+part,buf_vectorize(df_cb))
np.save(loc+'/sb'+part,buf_vectorize(df_sb))
print 'done'

#total_model.fit(X,Y)
#total_model.evaluate(X,Y)
#total_model.predict(X)
