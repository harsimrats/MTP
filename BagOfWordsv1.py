from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
import numpy as np
from ListOfStrings import ListOfStrings
class BagOfWordsv1:
    def __init__(self, listOfStrings, mingram, maxgram):
        self.vectorizer = CountVectorizer()
        self.mingram = mingram
        self.maxgram = maxgram
        listOfSentences, self.maxlen = getSentences(listOfStrings, mingram, maxgram)
        baggedData = self.vectorizer.fit_transform(listOfSentences)
        self.tfidf = TfidfTransformer().fit(baggedData)
    def process(self, listOfStrings):
        listOfSentences, _ = getSentences(listOfStrings,self.mingram, self.maxgram, maxlen=self.maxlen)
        baggedData = self.vectorizer.transform(listOfSentences)
        baggedData = self.tfidf.transform(baggedData)
        return baggedData

def getSentences(listOfStrings,mingram, maxgram, maxlen=None):
    listOfStringstemp = removeHash(listOfStrings)
    # print(listOfStringstemp)
    listOfStringstemp = ListOfStrings(listOfStringstemp)
    if maxlen is not None:
        listOfStringstemp.maxlen = maxlen
    listOfListOfStringstemp = ngramize(mingram, maxgram, listOfStringstemp)
    # print(listOfListOfStringstemp)
    listOfSentences = np.array([" ".join(x) for x in listOfListOfStringstemp])
    return listOfSentences, listOfStringstemp.maxlen

def removeHash(listOfStrings):
    return np.array([":".join(x.split(":")[1:]) for x in listOfStrings])

def ngramize(mingram, maxgram, listOfStrings):
    ngramData = np.array([])
    # print("maxlen is ", listOfStrings.maxlen)
    # print(listOfStrings.maxlen)
    for gramsize in range(mingram, min(maxgram,listOfStrings.maxlen)+1):
        gramData = gramize(listOfStrings, gramsize)
        if ngramData.shape[0] == 0:
            ngramData = gramData.copy()
        else:
            ngramData = np.append(ngramData, gramData, axis=1)
    return ngramData

def gramize(listOfStrings, gramsize):
    X = listOfStrings.listOfStrings
    numele = len(X)
    reqdim = listOfStrings.maxlen - gramsize +1
    Y = []
    for i in X:
        tempy = []
        paddedi = i + "0"*(listOfStrings.maxlen-len(i))
        for j in range(0,len(paddedi)-gramsize+1):
            tempy.append(paddedi[j:j+gramsize])
        if len(tempy) == 0:
            tempy = [paddedi]
        templeny = len(tempy)
        Y.append(tempy)
    return np.array(Y)