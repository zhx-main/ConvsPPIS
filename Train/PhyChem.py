import csv
import os
import sys
os.chdir('D:\workshop\PythonWorkspace\PPIs/')
from keras.callbacks import ModelCheckpoint
from sklearn.model_selection import StratifiedKFold, train_test_split

from Datas.Datasets import Datasets
from Model.CNN.CnnModel import CNNModel
from Pipeline.util.tools import calculate_performace
from Result.Train.EvalModel import EvalModel

def trainPhychem():
    f = open(r'Result/Train/train10fold.csv', 'w', newline='')
    writer = csv.writer(f)
    writer.writerows([['len', 'acc', 'prec', 'recall', 'spec', 'f1', 'mcc','auc_roc','auc_pr']])
    f.flush()
    f.close()
    WINSIZE=8
    winsizelist=[5,6,7,8,9,10]

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
            model.save('Result/Train/DifferentRadius/phychem/phychem'+str(size)+'_'+ str(modelIdx) + '.h5')
            y_pre=model.predict(x_test)
            p_classes = [int(prob > 0.5) for prob in y_pre.flatten()]
            eval.evalModel(size,y_pre,p_classes, y_test)
        #eval.printcrosseval()
        eval.evalTocsv()
def trainPhychemDNN():
    f = open(r'Result/Train/train10fold.csv', 'w', newline='')
    writer = csv.writer(f)
    writer.writerows([['len', 'acc', 'prec', 'recall', 'spec', 'f1', 'mcc', 'auc_roc', 'auc_pr']])
    f.flush()
    f.close()
    WINSIZE = 8
    data, label = Datasets.getPhyChem('DBv5', WINSIZE)
    data = data.reshape(data.shape[0], data.shape[1] * data.shape[2])
    skf = StratifiedKFold(n_splits=10, random_state=0, shuffle=True)
    eval = EvalModel()
    modelIdx = 0
    for train, test in skf.split(data, label):
        x_train = data[train]
        x_test = data[test]
        y_train = label[train]
        y_test = label[test]
        model = CNNModel.DNN(shape=(WINSIZE * 2 + 1, 10))
        modelIdx += 1
        model.fit(x_train, y_train, batch_size=100, epochs=100,
                  verbose=2)
        model.save('Result/Train/CompaDNN/phychem/phychem' + str(WINSIZE) + '_' + str(modelIdx) + '.h5')
        y_pre = model.predict(x_test)
        p_classes = [int(prob > 0.5) for prob in y_pre.flatten()]
        eval.evalModel(WINSIZE, y_pre, p_classes, y_test)
    eval.evalTocsv()
if __name__ == '__main__':
    trainPhychemDNN()