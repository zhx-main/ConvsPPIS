from Datas.Feature.Pssm.Pssm import Pssm
from Datas.Feature.PhyChem.PhyChem import PhyChem
from Datas.Feature.PSAIA.PSAIA import PSAIA
from Datas.Feature.BFactor.BFactor import BFactor
import numpy as np
np.random.seed(1)

class Datasets():
    def __init__(self):
        pass
    @staticmethod
    def shuffleData(data,label):
        i = [i for i in range(len(label))]
        a=np.array(i)
        np.random.seed(1)
        np.random.shuffle(i)
        return np.array(data)[i],np.array(label)[i]

    @staticmethod
    def getPSSM(fname,size=4):
        data,label=Pssm(fname).PssmWinsize(size,False,True)
        data,label=Datasets.shuffleData(data,label)
        return (data,label)
    @staticmethod
    def narrowNegSample(fname,size=4):
        data,label=Pssm(fname).negSampControl(size)
        data,label=Datasets.shuffleData(data,label)
        return (data,label)
    @staticmethod
    def getPSSMWithBNotSacle(fname,size=4):
        data,label=Pssm(fname).PssmWinsize(size,True,False)
        data,label=Datasets.shuffleData(data,label)
        return (data,label)
    @staticmethod
    def getPSSMWithoutBNotScale(fname,size=4):
        data,label=Pssm(fname).PssmWinsize(size,False,False)
        data,label=Datasets.shuffleData(data,label)
        return (data,label)

    @staticmethod
    def getPSSMWithBScale(fname, size=4):
        data, label = Pssm(fname).PssmWinsize(size, True, True)
        data, label = Datasets.shuffleData(data, label)
        return (data, label)
#-------------------------------------------------------------------
    @staticmethod
    def getPhyChem(fname,size):
        data,label=PhyChem(fname).phyChemWinsize(size=size)
        data,label=Datasets.shuffleData(data,label)
        return (data,label)

    @staticmethod
    def getPhyChemWithB(fname, size):
        data, label = PhyChem(fname).phyChemWinsize(size=size,withB=True)
        data, label = Datasets.shuffleData(data, label)
        return (data, label)

#-------------------------------------------------------------------
    @staticmethod
    def getPSAIA(fname,size):
        data,label=PSAIA(fname).psaiaWinsize(size)
        data,label=Datasets.shuffleData(data,label)
        return (data,label)

    @staticmethod
    def getPSAIAWithB(fname, size,withB=True):
        data, label = PSAIA(fname).psaiaWinsize(size,withB=withB)
        data, label = Datasets.shuffleData(data, label)
        return (data, label)

#-------------------------------------------------------------------
    @staticmethod
    def getConsPhychem(fname,size):
        data_pssm,label_pssm=Datasets.getPSSM(fname,size)
        data_phychem,label_phychem=Datasets.getPhyChem(fname,size)
        data=np.concatenate([data_pssm,data_phychem],axis=-1)
        label=np.concatenate([label_pssm,label_phychem],axis=-1)
        return (data,label)

