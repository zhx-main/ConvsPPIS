import csv

from keras.callbacks import Callback
import numpy as np
from keras.engine.saving import load_model
from keras.utils import np_utils
from Datas.Datasets import Datasets
from Model.CNN.CnnModel import CNNModel
from Pipeline.cnn.Alone.Metricss import Metricss
from Pipeline.util.tools import calculate_performace
from Result.Alone.EvalModel import EvalModel


class AloneTest(Metricss):

    def __init__(self):
        pass

    @staticmethod
    def train_model():
        winlist = [8]
        for w in winlist:

            pssm_train,pssm_trainlabel=Datasets.getPSSM('DBv5',size=w)
            phychem_train,phychem_trainlabel=Datasets.getPhyChem('DBv5',size=w)
            psaia_train,psaia_trainlabel=Datasets.getPSAIA('DBv5',size=w)
            pssm_test, pssm_testlabel = Datasets.getPSSM('CAPRI', size=w)
            phychem_test,phychem_testlabel=Datasets.getPhyChem('CAPRI',size=w)
            psaia_test,psaia_testlabel=Datasets.getPSAIA('CAPRI',size=w)
            trainData=np.concatenate([pssm_train,phychem_train,psaia_train],axis=-1)
            testData=np.concatenate([pssm_test,phychem_test,psaia_test],axis=-1)
            # pssm_train =pssm_train.reshape(((-1,17,20,1)))
            # pssm_test = pssm_test.reshape(((-1, 17, 20, 1)))
    
            model=CNNModel.CNN1D(shape=(17,20))
            me=Metricss()
            model.fit(pssm_train,pssm_trainlabel,batch_size=50,epochs=400,
                      verbose=1,
                      callbacks=[me])
    @staticmethod
    def load_model_test():
        w=8
        test_data, test_label = Datasets.getPSSM('CAPRI', size=w)
        eval=EvalModel(8)
        model=load_model('trainedmodel directory')
        y_pro=model.predict(test_data)
        y_class = [int(prob > 0.5) for prob in y_pro.flatten()]
        tp, fp, tn, fn, accuracy, precision, sensitivity, specificity, MCC, f1_score = \
            calculate_performace(len(y_class), y_class, test_label)
        print('acc:{},pre:{},rec:{},f1:{},MCC:{}'.format(accuracy,precision,sensitivity,f1_score,MCC))

if __name__ == '__main__':
    AloneTest.load_model_test()

