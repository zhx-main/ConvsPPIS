ConvsPPIS: Identifying Protein-protein Interaction Sites by an Ensemble Convolutional Neural Network with Feauture Graph
=======
## Directory
### -Datas
  The protein sequence of DBv5-Sel and CAPRI-Alone and its corresponding lables. 
### -Model
  The model architecture we used in our expriments: CNN(Convolutional neural network) and FCN(Fully-Connected Network).
### -Train
  The Pipeline of training process with 10-fold cross validation for PSSM, PhyChem, and PSAIA respectively.  
### -DifferentRadius
  The trained model of different sliding window size (including the responding radius 5,6,7,8,9,10) with 10-fold cross validation.<br>
<br>
## Enviorment<br>
 * Python 3.5 <br>
 * scikit-learn 0.20.1 <br>
 * tensorflow 1.9.0<br>
 * keras 2.2.4<br>
