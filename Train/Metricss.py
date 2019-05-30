from keras.callbacks import Callback
import numpy as np
from Pipeline.util.tools import calculate_performace


class Metricss(Callback):
    def __init__(self):
        self.path = 'E:\PythonWorkspace\PPIs\Result\Train\AlonePerEpoch.csv'
        super(Metricss, self).__init__()

    def on_train_begin(self, logs=None):
        with open(self.path, 'a+') as f:
            f.writelines('epoch,tp,fp,tn,fn,accuracy, precision, recall, specificity,f1_score, MCC' + '\n')
            f.close()
    def on_epoch_end(self, epoch, logs=None):
        path='E:\modelsDirectory\\'
        model=self.model
        test_data=self.validation_data[0]
        test_label=self.validation_data[1]
        y_pre=model.predict(test_data)
        #y_pre=np.argmax(y_pre,axis=1)
        model.save(path+'model_'+str(test_data.shape[1])+'_'+str(epoch+1)+'.h5')
        y_pre = [int(prob > 0.5) for prob in y_pre.flatten()]
        tp, fp, tn, fn, accuracy, precision, sensitivity, specificity, MCC, f1_score=\
            calculate_performace(len(y_pre),y_pre,test_label)
        with open(self.path,'a+') as f:
            f.writelines(str(epoch+1)+","+str(tp)+","+str(fp)+","+str(tn)+","+str(fn)+","+str(accuracy)+","+str(precision)+","+ \
                        str(sensitivity)+","+str(specificity)+","+str(f1_score)+","+str(MCC)+'\n')
