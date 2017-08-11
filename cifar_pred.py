import keras
from keras.models import load_model
from keras.datasets import cifar10
import os

batch_size = 32
num_classes = 10

(x_train, y_train), (x_test, y_test) = cifar10.load_data()

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



backList = ["cntk", "tensorflow", "theano"]
saveDir = "/tmp/nfstest/cifar_model/"
files = os.listdir(saveDir)
filesList = [f for f in files if os.path.isfile(os.path.join(saveDir, f))]
filesList


prediction = 0
sumAcc = 0
for bk in backList:
    os.environ["KERAS_BACKEND"] = bk
    kerasBKED = os.environ["KERAS_BACKEND"]
    modelName = [m for m in filesList if bk in m][0]
    modelAcc = float(modelName.split("_")[1])
    print("MODELNAME {0} \t for BACKEND {1} \t with ACCURACY {2}".format(modelName, kerasBKED, modelAcc))
    model = load_model(os.path.join(saveDir, modelName))
    prediction += model.predict(x_test2, batch_size=batch_size, verbose=0) * modelAcc
    sumAcc += modelAcc
prediction = prediction / sumAcc
print(prediction)


import csv
with open("/tmp/nfstest/cifar_model/preds.csv", 'w') as f:
    writer = csv.writer(f, lineterminator='\n')
    writer.writerow([0,1,2,3,4,5,6,7,8,9])
    writer.writerows(prediction)  


predlist = []
for i in prediction:
    pred = max(enumerate(i), key=lambda x: x[1])[0]
    predlist.append([pred, i[pred]])
print(predlist)


with open("/tmp/nfstest/cifar_model/pred.csv", 'w') as f:
    writer = csv.writer(f, lineterminator='\n')
    writer.writerow(["prediction","rate"])
    writer.writerows(predlist)  
