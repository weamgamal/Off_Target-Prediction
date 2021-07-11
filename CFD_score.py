
from CFD_Scoring_1 import*


def loadData(inputpath):
   
    with open(inputpath) as f:
        sgRNA_item=[]
        DNA_item=[]
        for line in f:
            ll = [i for i in line.strip().split(',')]
            sgRNA_item.append(ll[0])
            DNA_item.append(ll[1])
    return DNA_item,sgRNA_item

glove_inputpath = "output\keras_GloVeVec_5_100_10000.csv"
hek_inputpath = "output\hek293_off_Glove.txt"
K562_inputpath = "output\K562_off_Glove.txt"
DNA_item,sgRNA_item = loadData(hek_inputpath)
f=0 

s=[]
while f<len(sgRNA_item):
    off =DNA_item[f]
    s.append(off)
    #print(off)
    #s+=off
    wt = sgRNA_item[f]
    s.append(wt)
    #print(wt)
    #s+=wt
    m_wt = re.search('[^ATCG]',str(wt))
    m_off = re.search('[^ATCG]',str(off))
    if (m_wt is None) and (m_off is None):

        pam = off[-2:]       
        sg = off[:-3]         
        cfd_score = calc_cfd(wt,sg,pam)
        s.append(cfd_score)
        #print(cfd_score)
       #s.append("####################################")
        s.append("_____________________________________")
        #s+= wt+off+"####"+str(cfd_score)+"____"
        f=f+1
print(s)
def retu():
    return s

