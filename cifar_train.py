import os
os.environ["KERAS_BACKEND"] = "cntk"
kerasBKED = os.environ["KERAS_BACKEND"]
print(kerasBKED)


import keras
from keras.models import load_model
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.callbacks import EarlyStopping, ModelCheckpoint
import os
import pickle
import numpy as np


batch_size = 32
num_classes = 10
epochs = 50


(x_train, y_train), (x_test, y_test) = cifar10.load_data()
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)


x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255


x_test1 = x_test[100:,:]
x_test2 = x_test[:100,:]
y_test1 = y_test[100:,:]
y_test2 = y_test[:100,:]
print(x_test1.shape, x_test2.shape, y_test1.shape, y_test2.shape)


model = Sequential()

model.add(Conv2D(32, (3, 3), padding='same',
                 input_shape=x_train.shape[1:]))
model.add(Activation('relu'))
model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(64, (3, 3), padding='same'))
model.add(Activation('relu'))
model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes))
model.add(Activation('softmax'))

opt = keras.optimizers.adam()

model.compile(loss='categorical_crossentropy',
              optimizer=opt,
              metrics=['accuracy'])



chkpt = '/tmp/nfstest/cifar_weights/' + kerasBKED + '_weights.{epoch:02d}-{loss:.2f}-{val_loss:.2f}.hdf5'
cp_cb = ModelCheckpoint(filepath = chkpt, monitor='val_loss', verbose=1, save_best_only=True, mode='auto')
es_cb = EarlyStopping(monitor='val_loss', patience=1, verbose=1, mode='auto')

history = model.fit(x_train, y_train,
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=1,
                    validation_data=(x_test1, y_test1),
                    callbacks=[cp_cb, es_cb],
                    shuffle=True)

score = model.evaluate(x_test1, y_test1, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

saveDir = "/tmp/nfstest/cifar_model/"
if not os.path.isdir(saveDir):
    os.makedirs(saveDir)
modelName = "{0}_{1}_model.hdf5".format(kerasBKED, score[1])
model_path = os.path.join(saveDir, modelName)
model.save(model_path)

