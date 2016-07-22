from keras.models import Sequential
from keras.layers import Dense, Activation

l1sb = Sequential()
l1cb = Sequential()

l2cip = Sequential()
l2sip = Sequential()
l2cp = Sequential()
l2sp = Sequential()
l2sb = Sequential()
l2cb = Sequential()

l3fc = Sequential()
l4fc = Sequential()
l5fc = Sequential()
l5tr = Sequential()

model = Sequential()
model.add(Dense(output_dim=64, input_dim=100))
model.add(Activation("relu"))
model.add(Dense(output_dim=10))
model.add(Activation("softmax"))

model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])
