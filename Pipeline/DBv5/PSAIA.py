import csv
import os
import sys
#os.chdir('root directory')
#sys.path.append('root directory')
from sklearn.model_selection import StratifiedKFold, train_test_split
from Datas.Datasets import Datasets
from Model.CNN.CnnModel import CNNModel
from Result.Train.EvalModel import EvalModel


def trainPSAIA():
    #winsizelist = [5, 6, 7, 8, 9, 10]
    winsizelist = [8]
    for size in winsizelist:
        data, label = Datasets.getPSAIA('DBv5', size)
        data = data[:,:,5:]
        skf=StratifiedKFold(n_splits=10,random_state=0,shuffle=True)
        modelIdx=0
        for train,test in skf.split(data,label):
            x_train=data[train];x_test=data[test]
            y_train=label[train];y_test=label[test]
            model=CNNModel.CNN1D(shape=(size*2+1,10))

            modelIdx+=1
            model.fit(x_train,y_train,batch_size=100,epochs=100,
                      verbose=2
                     )
            #model.save(''+str(size)+'_'+ str(modelIdx)+'.h5')
            y_pre=model.predict(x_test)
            p_classes = [int(prob > 0.5) for prob in y_pre.flatten()]
            eval.evalModel(size,y_pre,p_classes, y_test)
        eval.evalTocsv()
