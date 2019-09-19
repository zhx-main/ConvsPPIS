import os
#os.chdir('root directory')
from sklearn.model_selection import StratifiedKFold, train_test_split

from Datas.Datasets import Datasets
from Model.CNN.CnnModel import CNNModel
from Result.Train.EvalModel import EvalModel

def trainPhychem():
    #winsizelist=[5,6,7,8,9,10]
    winsizelist = [8]  # the window size is 8*2+1=17
    for size in winsizelist:
        data, label = Datasets.getPhyChem('DBv5', size)
        skf=StratifiedKFold(n_splits=10,random_state=0,shuffle=True)
        eval=EvalModel()
        modelIdx=0
        for train,test in skf.split(data,label):
            x_train=data[train];x_test=data[test]
            y_train=label[train];y_test=label[test]
            model=CNNModel.CNN1D(shape=(size*2+1,10))
            modelIdx+=1
            model.fit(x_train,y_train,batch_size=100,epochs=100,
                      verbose=2)
            y_pre=model.predict(x_test)
            p_classes = [int(prob > 0.5) for prob in y_pre.flatten()]
            eval.evalModel(size,y_pre,p_classes, y_test)
        eval.evalTocsv()
