from keras.engine.saving import load_model
from sklearn.model_selection import StratifiedKFold
from Datas.Datasets import Datasets
from Pipeline.util.tools import calculate_performace
import numpy as np
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import average_precision_score
import os
import sys
os.chdir('root directory')
sys.path.append('root directory')
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels


def plot_confusion_matrix(y_true, y_pred, classes,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """

    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
    classes = classes[unique_labels(y_true, y_pred)]
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return ax


colors = ['dodgerblue','orange','chartreuse','plum','burlywood',
          'saddlebrown','green','purple','rosybrown','yellowgreen']

WINSIZE=8
pssm_data,pssm_label=Datasets.getPSSM('DBv5',size=WINSIZE)
phychem_data,phychem_label=Datasets.getPhyChem('DBv5',WINSIZE)
psaia_data,psaia_label=Datasets.getPSAIA('DBv5',WINSIZE)
skf=StratifiedKFold(n_splits=10,random_state=0,shuffle=True)
modelIdx=1
fold10=[]
pssmFold10=[]
phychemFold10=[]
psaiaFold10=[]
plt.figure(figsize=(10, 10))
for train, test in skf.split(pssm_data, pssm_label):
        pssm_data_test=pssm_data[test];pssm_label_test=pssm_label[test]
        phychem_data_test=phychem_data[test];phychem_label_test=phychem_label[test]
        psaia_data_test=psaia_data[test];psaia_label_test=psaia_label[test]

# ======================================pssm=============================================================================
        pssm_model=load_model('trainedModel directory/' + str(modelIdx) + '.h5')
        pssm_pre = pssm_model.predict(pssm_data_test)
        y_classes = [int(prob > 0.5) for prob in pssm_pre.flatten()]
        tp, fp, tn, fn, accuracy, precision, sensitivity, specificity, MCC, f1_score = \
            calculate_performace(len(y_classes), y_classes, pssm_label_test)
        pssmFold10.append([accuracy,precision,sensitivity,f1_score,MCC])
        # print('acc:{},pre:{},rec:{},f1:{},MCC:{}'.format(accuracy, precision, sensitivity, f1_score, MCC))
# # ======================================phychem=============================================================================
        phychem_model = load_model('trainedModel directory/' + str(modelIdx) + '.h5')
        phychem_pre = phychem_model.predict(phychem_data_test)
        y_classes = [int(prob > 0.5) for prob in phychem_pre.flatten()]
        tp, fp, tn, fn, accuracy, precision, sensitivity, specificity, MCC, f1_score = \
            calculate_performace(len(y_classes), y_classes, phychem_label_test)
        phychemFold10.append([accuracy,precision,sensitivity,f1_score,MCC])
        # print('acc:{},pre:{},rec:{},f1:{},MCC:{}'.format(accuracy, precision, sensitivity, f1_score, MCC))

#======================================psaia=============================================================================
        psaia_model=load_model('trainedModel directory/'+str(modelIdx)+'.h5')
        psaia_pre=psaia_model.predict(psaia_data_test)
        y_classes = [int(prob > 0.5) for prob in psaia_pre.flatten()]
        tp, fp, tn, fn, accuracy, precision, sensitivity, specificity, MCC, f1_score = \
            calculate_performace(len(y_classes), y_classes, psaia_label_test)
        psaiaFold10.append([accuracy,precision,sensitivity,f1_score,MCC])
        # print('acc:{},pre:{},rec:{},f1:{},MCC:{}'.format(accuracy, precision, sensitivity, f1_score, MCC))
# ======================================concat=============================================================================
        con_pre=np.concatenate([pssm_pre,phychem_pre,psaia_pre],axis=1)
        y_pre=np.average(con_pre,axis=1)
        y_classes = [int(prob > 0.465) for prob in y_pre.flatten()]
        tp, fp, tn, fn, accuracy, precision, sensitivity, specificity, MCC, f1_score = \
            calculate_performace(len(y_classes), y_classes, psaia_label_test)
        # print('acc:{},pre:{},rec:{},f1:{},MCC:{}'.format(accuracy, precision, sensitivity, f1_score, MCC))

        fpr, tpr, threshold = roc_curve(psaia_label_test, y_pre)
        roc_auc = auc(fpr, tpr)
        pr_auc = average_precision_score(psaia_label_test,y_pre)
        fold10.append([accuracy, precision, sensitivity,specificity, f1_score, MCC, roc_auc, pr_auc])
        lw = 2
        # plt2.plot(fpr, tpr, color=colors[modelIdx-1],
        #          lw=lw, label=str(modelIdx)+'-fold (area = %0.3f)' % roc_auc)
        modelIdx += 1
        print([accuracy, precision, sensitivity,specificity, f1_score, MCC, roc_auc, pr_auc])

        plot_confusion_matrix(psaia_label_test, np.array(y_classes,dtype='int'), classes=np.array(['non-interface','interface']),
                        title='confusion matrix')
        plt.show()

result=np.average(fold10,axis=0)
print(result)