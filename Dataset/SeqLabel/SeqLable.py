#数据的序列和标签
import os
import numpy as np
class SeqLable(object):
    def __init__(self,fname):
        self.proteins=[]
        self.noBProteins=[]
        self.fname=fname
        self.label = {'B': 2, 'I': 1, 'N': 0}
        self.getProteins(fname)
        self.getNoBProteins()
    def getProteins(self,fname):
        path='D:\workshop\PythonWorkspace\PPIs\Datas\SeqLabel/'+fname+'/'+fname+'.fa'
        with open(path,'r') as f:
            headID='';seq='';label=''
            for i,line in enumerate(f):
                if i%3==0:
                    if fname=='DBv5':
                        headID=line[1:7]
                    if fname=='CAPRI':
                        headID=line[1:-1]
                elif i%3==1:
                    seq=line.replace('\n','')
                else:
                    label=line.replace('\n','')
                    protein=(headID,seq,label)
                    self.proteins.append(protein)
    def getNoBProteins(self):
        for(id,seq,label) in self.proteins:
            seq=list(seq)
            label=list(label)
            if len(seq)!=len(label):
                print("SeqLabel中seq!=label")
            Idx=np.argwhere(np.array(label)!='B').flatten()
            seq=np.array(seq)[Idx]
            label=np.array(label)[Idx]
            seq=''.join(seq.tolist())
            label=''.join(label.tolist())
            self.noBProteins.append((id,seq,label))
    def statistics(self):
        """统计标签"""
        AAI={}
        proteins=self.proteins
        for id,seq,label in proteins:
            for i,(aa,la) in enumerate(zip(seq,label)):
                if la=='I':
                    if aa not in AAI.keys():
                        AAI[aa]=1
                    else:
                        AAI[aa]+=1
        print(sorted(AAI.items(), key=lambda x: x[1], reverse=True))



if __name__ == '__main__':
    SeqLable('CAPRI').statistics()




