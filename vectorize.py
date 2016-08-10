import pandas as pd
import numpy as np
import sys

loc = 'binary/splited/'
part = ''

threshold = 8 # KB

split_bin = 100000

def vector(value,size):
	v = [0]*size
	v[value] = 1
	return v

def ip_vectorize(df_col):
	v = map(lambda x: map(lambda y: vector(int(y), 256), x.split('.')), df_col.values)
	return np.array(v).reshape((len(v), 256*4))

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

def port_vectorize(df_col, array):
	v_size = len(array)
	v = map(lambda x: vector(array.index(x), v_size) if (x < 1024) else vector(array.index(1024), v_size), df_col.values)
	return np.array(v)

#def port_vectorize(df_col):
#	v = map(lambda x: vector(x+1, 1025) if (x < 1024) else vector(0, 1025), df_col.values)
#	print np.array(v).shape
#	return np.array(v)

def buf_vectorize(df_col):
	v = map(lambda x: map(lambda y: vector(y+1, 257), x), df_col.values)
	return np.array(v).reshape((len(v), 1, 16, 257))

def truth_vectorize(df_col):
	global threshold
	v = map(lambda x: vector(x, 2), ((df_col >= (threshold*1024)) + 0).values)
	return np.array(v)

def write_to_npy(name, np_array):
	print name + ': ',
	print np_array.shape
	np.save(loc+name, np_array)

print 'Reading flow data...'
df_total = pd.read_csv('flowData/flow_dump.csv')
df_cb_total = pd.read_csv('flowData/flow_dump_cb.csv').drop('dummy',axis=1)
df_sb_total = pd.read_csv('flowData/flow_dump_sb.csv').drop('dummy',axis=1)

print 'Vectorizing data...'
print '(Threshold: ' + str(threshold) + 'KB, ',
print 'split_bin: ' + str(split_bin) + ')'

cp_list = comp_port(df_total.cp)
sp_list = comp_port(df_total.sp)

i = 0
df_len = len(df_total)

while i < df_len:
	j = min(df_len, i + split_bin)
	df = df_total[i:j]
	df_cb = df_cb_total[i:j]
	df_sb = df_sb_total[i:j]
	part = '_' + str(i / split_bin)
	i = j
	write_to_npy('cip'+part,ip_vectorize(df.cip))
	write_to_npy('sip'+part,ip_vectorize(df.sip))
	write_to_npy('cp'+part, port_vectorize(df.cp, cp_list))
	write_to_npy('sp'+part, port_vectorize(df.sp, sp_list))
	write_to_npy('cb'+part, buf_vectorize(df_cb))
	write_to_npy('sb'+part, buf_vectorize(df_sb))
	write_to_npy('tr_'+str(threshold)+'kb'+part, truth_vectorize(df.flow_size))

#	np.save(loc+'/cip'+part,ip_vectorize(df.cip))
#	np.save(loc+'/sip'+part,ip_vectorize(df.sip))
#	np.save(loc+'/cp'+part,port_vectorize(df.cp, cp_list))
#	np.save(loc+'/sp'+part,port_vectorize(df.sp, sp_list))
#	np.save(loc+'/cb'+part,buf_vectorize(df_cb))
#	np.save(loc+'/sb'+part,buf_vectorize(df_sb))
#	np.save(loc+'/tr_'+str(threshold)+'kb'+part,truth_vectorize(df.flow_size))
print 'done'
