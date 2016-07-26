from keras.models import Sequential
from keras.layers import Dense, Merge, Activation

FCDIM = 128
FTSIZE = 4
BUFLEN = 16
numft = 4

layer_sb = Sequential([
		Convolution2D(numft, FTSIZE, 256, border_mode='???', input_shape=(1, BUFLEN, 256)),
		Activation('relu'),
		Dense(FCDIM)
])
layer_cb = Sequential([
		Convolution2D(numft, FTSIZE, 256, border_mode='???', input_shape=(1, BUFLEN, 256)),
		Activation('relu'),
		Dense(FCDIM)
])

layer_cip = Sequential([Dense(FCDIM, input_dim=(4*256))])
layer_sip = Sequential([Dense(FCDIM, input_dim=(4*256))])
layer_cp = Sequential([Dense(FCDIM, input_dim=1024)])
layer_sp = Sequential([Dense(FCDIM, input_dim=1024)])

layer_merged = Merge([layer_cip,layer_sip,layer_cp,layer_sp,layer_sb,layer_cb],
		mode = 'sum')

total_model = Sequential()
total_model.add(layer_merged)
total_model.add(Activation('relu'))
total_model.add(Dense(2))
total_model.add(Activation('relu'))

total_model.compile(loss='binary_crossentropy', optimizer='sgd', metrics=['accuracy'])

#total_model.fit(X,Y)
#total_model.evaluate(X,Y)
#total_model.predict(X)
