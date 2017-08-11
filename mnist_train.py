import os
#os.environ["KERAS_BACKEND"] = "tensorflow"
kerasBKED = os.environ["KERAS_BACKEND"] 
print(kerasBKED)


import keras
from keras.models import load_model
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import RMSprop
from keras.callbacks import EarlyStopping, ModelCheckpoint


batch_size = 128
num_classes = 10
epochs = 10

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.reshape(60000, 784)
x_test = x_test.reshape(10000, 784)
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)


x_test1 = x_test[10:,:]
x_test2 = x_test[:10,:]
y_test1 = y_test[10:,:]
y_test2 = y_test[:10,:]
x_test1.shape, x_test2.shape, y_test1.shape, y_test2.shape


model = Sequential()
model.add(Dense(512, activation='relu', input_shape=(784,)))
model.add(Dropout(0.2))
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(10, activation='softmax'))

model.summary()

model.compile(loss='categorical_crossentropy',
              optimizer=RMSprop(),
              metrics=['accuracy'])


chkpt = '/tmp/nfstest/mnist_weights/' + kerasBKED + '_weights.{epoch:02d}-{loss:.2f}-{val_loss:.2f}.hdf5'
cp_cb = ModelCheckpoint(filepath = chkpt, monitor='val_loss', verbose=1, save_best_only=True, mode='auto')
es_cb = EarlyStopping(monitor='val_loss', patience=2, verbose=1, mode='auto')


history = model.fit(x_train, y_train,
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=1,
                    validation_data=(x_test, y_test),
                    callbacks=[cp_cb, es_cb])


score = model.evaluate(x_test1, y_test1, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])


saveDir = "/tmp/nfstest/mnist_model"
if not os.path.isdir(saveDir):
    os.makedirs(saveDir)
modelName = "{0}_{1}_model.hdf5".format(kerasBKED, score[1])
model_path = os.path.join(saveDir, modelName)
model.save(model_path)


