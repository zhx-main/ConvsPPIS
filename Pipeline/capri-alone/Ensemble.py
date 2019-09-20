from keras.engine.saving import load_model
import numpy as np
from sklearn.utils.multiclass import unique_labels
import os
import sys
#os.chdir('root directory')
#sys.path.append('root directory')
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import average_precision_score
from sklearn.metrics import confusion_matrix
from Datas.Datasets import Datasets
from Pipeline.util.tools import calculate_performace
from Result.Alone.EvalModel import EvalModel
import matplotlib.pyplot as plt
eval=EvalModel(8)
w=8
pssm_data,pssm_label = Datasets.getPSSM('CAPRI', size=w)
phychem_data,phychem_label=Datasets.getPhyChem('CAPRI',size=w)
psaia_data,psaia_label=Datasets.getPSAIA('CAPRI',size=w)

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

pssmModel = load_model('trainedmodel directory ')
phychemModel = load_model('trainedmodel directory')
psaiaModel = load_model('trainedmodel diretory')

m1_pro=pssmModel.predict(pssm_data)
pssm_class = [int(prob > 0.5) for prob in m1_pro.flatten()]
tp, fp, tn, fn, accuracy, precision, sensitivity, specificity, MCC, f1_score = \
            calculate_performace(len(pssm_class), pssm_class, pssm_label)
print('acc:{},pre:{},rec:{},f1:{},MCC:{}'.format(accuracy,precision,sensitivity,f1_score,MCC))

m2_pro=phychemModel.predict(phychem_data)
phychem_class = [int(prob > 0.5) for prob in m2_pro.flatten()]
tp, fp, tn, fn, accuracy, precision, sensitivity, specificity, MCC, f1_score = \
            calculate_performace(len(phychem_class), phychem_class, phychem_label)
print('acc:{},pre:{},rec:{},f1:{},MCC:{}'.format(accuracy,precision,sensitivity,f1_score,MCC))

m3_pro=psaiaModel.predict(psaia_data)
psaia_class = [int(prob > 0.5) for prob in m3_pro.flatten()]
tp, fp, tn, fn, accuracy, precision, sensitivity, specificity, MCC, f1_score = \
            calculate_performace(len(psaia_class), psaia_class, psaia_label)
print('acc:{},pre:{},rec:{},f1:{},MCC:{}'.format(accuracy,precision,sensitivity,f1_score,MCC))

c=np.concatenate([m1_pro,m2_pro,m3_pro],axis=1)
c_pro=np.average(c,axis=1)
y_class = [int(prob > 0.465) for prob in c_pro.flatten()]
eval.evalSingleModel(c_pro,y_class,pssm_label)
tp, fp, tn, fn, accuracy, precision, sensitivity, specificity, MCC, f1_score = \
            calculate_performace(len(y_class), y_class, pssm_label)
print('acc:{},pre:{},rec:{},f1:{},MCC:{}'.format(accuracy,precision,sensitivity,f1_score,MCC))
fpr, tpr, threshold = roc_curve(pssm_label, c_pro)
roc_auc = auc(fpr, tpr)
pr_auc = average_precision_score(pssm_label, c_pro)
print(roc_auc)
print(pr_auc)
plot_confusion_matrix(pssm_label, np.array(y_class,dtype='int'), classes=np.array(['non-interface','interface']),
                              title='confusion matrix')
plt.show()
lw = 2
plt.plot(fpr, tpr, color='orange',
         lw=lw, label= 'ROC (area = %0.3f)' % roc_auc)  
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.show()