import pandas as pd
import numpy as np
import sys

loc = 'binary/split/compressed/'
part = ''

def vector(value,size):
	v = [0]*size
	v[value] = 1
	return v

def comp_ip(df_col):
	ips = []
	while len(df_col) > 0:
		ip = df_col.values[0]
		ips.append(ip)
		df_col = df_col[df_col != ip]
	return ips

def comp_port(df_col):
	ports = []
	while len(df_col) > 0:
		port = df_col.values[0]
		if int(port) >= 1024:
			ports.append(1024)
			df_col = df_col[df_col < 1024]
		else:
			ports.append(port)
			df_col = df_col[df_col != port]
	return ports

def vectorize(df_col, array):
	v_size = len(array)
	# v = map(lambda x: vector(array.index(x), v_size), df_col.values)
	v = map(lambda x: vector(array.index(x), v_size) if (x < 1024) else vector(array.index(1024), v_size), df_col.values)
	print np.array(v).shape
	return np.array(v)

def ip_vectorize(df_col):
	v = map(lambda x: map(lambda y: vector(int(y), 256), x.split('.')), df_col.values)
	print np.array(v).shape
	return np.array(v).reshape((len(v), 256*4))

def port_vectorize(df_col):
	v = map(lambda x: vector(x+1, 1025) if (x < 1024) else vector(0, 1025), df_col.values)
	print np.array(v).shape
	return np.array(v)

print 'Reading flow data...'
df = pd.read_csv('flowData/flow_dump.csv')[0:100000]
print 'Vectorizing compressed data...'

#cip = comp_ip(df.cip)
#print 'cip size: '+str(len(cip))
#sip = comp_ip(df.sip)
#print 'sip size: '+str(len(sip))
cp = comp_port(df.cp)
print 'cp size: '+str(len(cp))
sp = comp_port(df.sp)
print 'sp size: '+str(len(sp))
#np.save(loc+'/cip'+part,vectorize(df.cip, cip))
#np.save(loc+'/sip'+part,vectorize(df.sip, sip))
np.save(loc+'cp'+part,vectorize(df.cp, cp))
np.save(loc+'sp'+part,vectorize(df.sp, sp))
print 'done'
