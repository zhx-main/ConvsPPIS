import os
#os.chdir('your root directory')
import pickle
from Pipeline import tools
from Model.CNN.CnnModel import CNNModel
WINSIZE = 8
def trainPhychem():
    for i in range(10):
        phychem_train_data, phychem_train_label = pickle.load("/Dataset/DBv5/phychem/phychem-train-" + str(i + 1) + ".pkl")
        phychem_test_data, phychem_test_label = pickle.load("/Dataset/DBv5/phychem/phychem-test-" + str(i + 1) + ".pkl")
        model=CNNModel.CNN1D(shape=(WINSIZE*2+1,10))
        model.fit(phychem_train_data,phychem_train_label,batch_size=100,epochs=100,
                      verbose=2)
        y_pre=model.predict(phychem_test_data)
        y_class = [int(prob > 0.5) for prob in y_pre.flatten()]
        tp, fp, tn, fn, accuracy, precision, sensitivity, specificity, MCC, f1_score = \
            tools.calculate_performace(len(y_class), y_class, phychem_test_label)
        print(tp, fp, tn, fn, accuracy, precision, sensitivity, specificity, MCC, f1_score)
