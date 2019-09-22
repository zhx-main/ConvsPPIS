from keras.engine.saving import load_model
import os
import sys
import numpy as np
#os.chdir('root directory')
#sys.path.append('root directory')
from Pipeline import tools
import pickle
WINSIZE = 8

fold10=[]
for i in range(10):
    pssm_test_data, pssm_test_label = pickle.load("/Dataset/DBv5/pssm/pssm-train-" + str(i + 1) + ".pkl")
    phychem_test_data, phychem_test_label = pickle.load("/Dataset/DBv5/phychem/phychem-train-" + str(i + 1) + ".pkl")
    psaia_test_data, psaia_test_label = pickle.load("/Dataset/DBv5/psaia/psaia-train-" + str(i + 1) + ".pkl")
    pssmModel = load_model('/TrainedModels/DBv5/pssm_model'+str(i+1)+".h5")
    phychemModel = load_model('/TrainedModels/DBv5/phy_chem_model'+str(i+1)+".h5")
    psaiaModel = load_model('/TrainedModels/DBv5/psaia_model'+str(i+1)+".h5")

    m1_pro=pssmModel.predict(pssm_test_data)
    pssm_class = [int(prob > 0.5) for prob in m1_pro.flatten()]
    tp, fp, tn, fn, accuracy, precision, sensitivity, specificity, MCC, f1_score = \
                tools.calculate_performace(len(pssm_class), pssm_class, pssm_test_label)
    print('acc:{},pre:{},rec:{},f1:{},MCC:{}'.format(accuracy,precision,sensitivity,f1_score,MCC))

    m2_pro=phychemModel.predict(phychem_test_data)
    phychem_class = [int(prob > 0.5) for prob in m2_pro.flatten()]
    tp, fp, tn, fn, accuracy, precision, sensitivity, specificity, MCC, f1_score = \
                tools.calculate_performace(len(phychem_class), phychem_class, phychem_test_label)
    print('acc:{},pre:{},rec:{},f1:{},MCC:{}'.format(accuracy,precision,sensitivity,f1_score,MCC))

    m3_pro=psaiaModel.predict(psaia_test_data)
    psaia_class = [int(prob > 0.5) for prob in m3_pro.flatten()]
    tp, fp, tn, fn, accuracy, precision, sensitivity, specificity, MCC, f1_score = \
                tools.calculate_performace(len(psaia_class), psaia_class, psaia_test_label)
    print('acc:{},pre:{},rec:{},f1:{},MCC:{}'.format(accuracy,precision,sensitivity,f1_score,MCC))

    c=np.concatenate([m1_pro,m2_pro,m3_pro],axis=1)
    c_pro=np.average(c,axis=1)
    y_class = [int(prob > 0.465) for prob in c_pro.flatten()]
    tp, fp, tn, fn, accuracy, precision, sensitivity, specificity, MCC, f1_score = \
            tools.calculate_performace(len(y_class), y_class, pssm_test_label)
    fold10.append([accuracy, precision, sensitivity, specificity, f1_score, MCC])

result=np.average(fold10,axis=0)
print(result)