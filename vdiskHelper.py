import ngram as ngram
import glob
import numpy as np
import math
from ListOfStrings import ListOfStrings
from BagOfWordsv1 import BagOfWordsv1


def readFileByLines(filename):
    f = open(filename, 'r')
    lst = f.readlines()
    f.close()
#     print(lst)
    lst = list(map(str.rstrip, lst))
    return lst
def getData(mingram,maxgram,lessData=False):
    foldername = "metadata_maps/vdisk/*.db"
    files = sorted(glob.glob(foldername))
    ctr=0
    alltemptableDatastr = []
    ssTableDatastr = {}
    for fle in files:
        thisline = readFileByLines(fle)
        if lessData:
            if(len(thisline)>1000):
                continue
        # thisline = [x.strip('\n') for x in thisline]

        ssTableDatastr[ctr] = np.array(thisline)
        ctr+=1
        alltemptableDatastr += thisline
        print(ctr,len(thisline))
    # print(lines)
    alltemptableDatastr = np.array(alltemptableDatastr)
    # print(ssTableDatastr)
    # print(alltemptableDatastr)
    print("done Reading data from disk")
    
    bagOfWordsModel = BagOfWordsv1(alltemptableDatastr,mingram,maxgram)
    return (ssTableDatastr, alltemptableDatastr, bagOfWordsModel)
def get(mingram=4, maxgram=4, lessData=False):
    (ssTableDatastr, alltemptableDatastr, bagOfWordsModel) = getData(mingram, maxgram, lessData)
    return (ssTableDatastr,alltemptableDatastr, bagOfWordsModel)