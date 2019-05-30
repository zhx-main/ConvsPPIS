import csv

from keras.callbacks import ModelCheckpoint
from keras_preprocessing.image import ImageDataGenerator
from sklearn.model_selection import StratifiedKFold
import numpy as np
from Datas.Datasets import Datasets
from Model.CNN.CnnModel import CNNModel
from Result.Train.EvalModel import EvalModel

f = open(r'E:\PythonWorkspace\PPIs\Result\Train\train10fold.csv', 'w', newline='')
writer = csv.writer(f)
writer.writerows([['len', 'acc', 'prec', 'recall', 'spec', 'f1', 'mcc']])
f.flush()
f.close()
WINSIZE=8

data,label=Datasets.getPSSM('DBv5',size=WINSIZE)

data=np.reshape(data,(-1,17,20,1))

skf=StratifiedKFold(n_splits=10,random_state=0,shuffle=True)
eval=EvalModel(WINSIZE)
modelIdx=0
for train,test in skf.split(data,label):
    x_train=data[train];x_test=data[test]
    y_train=label[train];y_test=label[test]
    #datagen.fit(x_train)
    model=CNNModel.CNN_4(shape=data.shape[1:4])

    modelIdx+=1
    path='E:\PythonWorkspace\PPIs\Result\Train\Models\model'+str(modelIdx)+'.hdf5'
    checkpointer=ModelCheckpoint(filepath=path,save_best_only=True)
    #model.fit_generator(datagen.flow(x_train,y_train,batch_size=400),epochs=200)
    model.fit(x_train,y_train,batch_size=500,epochs=220,
              verbose=1,validation_data=(x_test,y_test),
              callbacks=[checkpointer])
    y_pre=model.predict(x_test)
    p_classes = [int(prob > 0.5) for prob in y_pre.flatten()]
    eval.evalModel(p_classes,y_test)
eval.printcrosseval()
eval.evalTocsv()

