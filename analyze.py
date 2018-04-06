import pandas as pd
import numpy as np

predict_total = np.load('log/dump/log_100_8k_160913.npy')
tr = []
for i in range(5):
	tr.append(np.load('binary/http/tr_8kb_'+str(i)+'.npy'))
truth = np.concatenate(tr, axis=0)
last_predict = [-1]*len(truth)
error = [0]*len(truth)
flip = [0]*len(truth)

for predict in predict_total:
	for i in range(len(predict)):
		if predict[i][0] > predict[i][1]: # predict 0
			if truth[i][1] == 1:
				error[i] += 1
			if last_predict[i] == 1:
				flip[i] += 1
			last_predict[i] = 0
		else: # predict 1
			if truth[i][0] == 1:
				error[i] += 1
			if last_predict[i] == 0:
				flip[i] += 1
			last_predict[i] = 1

a = open('log/dump/analyze.csv','w')
a.write('error,flip\n')
for i in range(len(truth)):
	a.write(str(error[i])+','+str(flip[i])+'\n')
a.close()