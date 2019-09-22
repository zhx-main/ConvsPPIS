import csv
# os.chdir('your directory')
# sys.path.append('your directory')
import pickle
from Pipeline import tools
from Model.CNN.CnnModel import CNNModel
def TrainPSSM():
    WINSIZE=8
    for i in range(10):
        pssm_train_data, pssm_train_label = pickle.load("/Dataset/DBv5/pssm/pssm-train-"+str(i + 1)+".pkl")
        pssm_test_data, pssm_test_label = pickle.load("/Dataset/DBv5/pssm/pssm-test-" + str(i + 1) + ".pkl")
        model=CNNModel.CNN1D(shape=(WINSIZE*2+1,20))
        model.fit(pssm_train_data,pssm_train_label,batch_size=100,epochs=100,
                  verbose=2)
        y_pre=model.predict(pssm_test_data)
        y_class = [int(prob > 0.5) for prob in y_pre.flatten()]
        tp, fp, tn, fn, accuracy, precision, sensitivity, specificity, MCC, f1_score=\
            tools.calculate_performace(len(y_class),y_class,pssm_test_label)
        print(tp, fp, tn, fn, accuracy, precision, sensitivity, specificity, MCC, f1_score)
