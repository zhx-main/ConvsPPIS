import csv
import os
import sys
#os.chdir('root directory')
#sys.path.append('root directory')
import pickle
from Pipeline import tools
from Model.CNN.CnnModel import CNNModel

WINSIZE =8
def trainPSAIA():
    for i in range(10):
        psaia_train_data, psaia_train_label = pickle.load("/Dataset/DBv5/psaia/psaia-train-" + str(i + 1) + ".pkl")
        psaia_test_data, psaia_test_label = pickle.load("/Dataset/DBv5/psaia/psaia-test-" + str(i + 1) + ".pkl")
        model = CNNModel.CNN1D(shape=(WINSIZE * 2 + 1, 10))
        model.fit(psaia_train_data, psaia_train_label, batch_size=100, epochs=100,
                  verbose=2)
        y_pre = model.predict(psaia_test_data)
        y_class = [int(prob > 0.5) for prob in y_pre.flatten()]
        tp, fp, tn, fn, accuracy, precision, sensitivity, specificity, MCC, f1_score = \
            tools.calculate_performace(len(y_class), y_class, psaia_test_label)
        print(tp, fp, tn, fn, accuracy, precision, sensitivity, specificity, MCC, f1_score)
