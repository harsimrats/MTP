#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import matplotlib.pyplot as plt
get_ipython().magic(u'matplotlib inline')
from pybloom import BloomFilter
import inspect


# In[ ]:


mu, sigma = 0, 0.1 # mean and standard deviation
s = np.random.normal(mu, sigma, 1000)


# In[ ]:


def generateGauss(numcompo, gap, sigma, numelepercompo):
    mu = 0
    ans = []
    for i in range(0, numcompo):
        s = np.random.normal(mu, sigma, numelepercompo)
#         print(s)
        ans += s.tolist()
        mu += gap
    return np.array([ans]).transpose()


# In[ ]:


def generateGaussArr(numcompo, gap, sigma, numelepercompo):
    mu = 0
    ans = []
    for i in range(0, numcompo):
        s = np.random.normal(mu, sigma, numelepercompo)
#         print(s)
        ans.append(s)
        mu += gap
    return np.array(ans)


# In[ ]:


s = generateGauss(2, 0.2, 0.1, 100)


# In[ ]:


s


# In[ ]:


sArr = generateGaussArr(2, 0.2, 0.1, 100)


# In[ ]:


np.random.shuffle(sArr)


# In[ ]:


np.mean(sArr[0])


# In[ ]:


sArr[0:2].flatten().reshape(-1,1)


# In[ ]:


print(s)


# In[ ]:


count, bins, ignored = plt.hist(s, 30, normed=True)
# plt.plot(bins, 1/(sigma * np.sqrt(2 * np.pi)) * np.exp( - (bins - mu)**2 / (2 * sigma**2) ), linewidth=2, color='r')
plt.show()


# In[ ]:


mean = [0, 0]
cov = [[1, 0], [0, 100]]  # diagonal covariance


# In[ ]:


x, y = np.random.multivariate_normal(mean, cov, 5000).T
plt.plot(x, y, 'x')
plt.axis('equal')
plt.show()


# In[ ]:


diag = np.identity(1)


# In[ ]:


diag = diag*0.01


# In[ ]:


diag


# In[ ]:


mu = np.zeros(1)


# In[ ]:


mu


# In[ ]:


val = np.random.multivariate_normal(mu,diag,100000).T


# In[ ]:


val


# In[ ]:


np.mean(val[0])


# In[ ]:


np.array([x,y,z])


# In[ ]:


l=[]
l.append(x).append(y).append(z)


# In[ ]:


l1=l.appe


# In[ ]:


datat = generateGauss(100, 0.4, 0.1, 100)


# In[ ]:


datat


# In[ ]:


plt.hist(datat.transpose()[0],500,normed=True)
plt.show()


# In[ ]:


trainIndi(datat,"1",100)


# In[ ]:


clf = models['1'][0]


# In[ ]:


tempans = []
for i in range(0,len(clf.means_)):
    thiscov = math.sqrt(clf.covariances_[i][0][0])
#     print(thiscov)
    thisweight = int(clf.weights_[i]*100*100)
#     print(thisweight)
    temps = np.random.normal(clf.means_[i],thiscov, thisweight)
    tempans += temps.tolist()
tempans = np.array([tempans]).transpose()
plt.hist(tempans.transpose()[0], 500, normed=True)


# In[ ]:


count=0
for i in datat.transpose()[0]:
    tempans = testIndi([i],'1')
    if(tempans==1):
        count +=1


# In[ ]:


print(count)
print(len(datat))


# In[ ]:


from sklearn import preprocessing
import numpy as np
import sys
from sklearn.metrics import accuracy_score
import pickle
from sklearn import mixture
import math

def ngram(X, gramsize):
    numele = len(X)
    minlen = len(X[0])
    maxlen = len(X[0])
    
    for i in X:
        templen = len(i)
        if(templen>maxlen):
            maxlen=templen
        if(templen<minlen):
            minlen=templen
    reqdim = maxlen - gramsize +1
    
    Y = []
    for i in X:
        tempy = []
        paddedi = i + "0"*(maxlen-len(i))
        for j in range(0,len(paddedi)-gramsize+1):
            tempy.append(paddedi[j:j+gramsize])
        if len(tempy) == 0:
            tempy = [paddedi]
        templeny = len(tempy)
        
        Y.append(tempy)
    return Y

def ngramEnforce(X, gramsize, reqdim):
    numele = len(X)
    minlen = len(X[0])
    maxlen = len(X[0])
    for i in X:
        templen = len(i)
        if(templen>maxlen):
            maxlen=templen
        if(templen<minlen):
            minlen=templen
    maxlen = reqdim + gramsize -1
    Y = []
    for i in X:
        tempy = []
        paddedi = i + "0"*(maxlen-len(i))
        for j in range(0,len(paddedi)-gramsize+1):
            tempy.append(paddedi[j:j+gramsize])
        templeny = len(tempy)
        
        Y.append(tempy)
    return np.array(Y)

def rstripfn(x):
    x = x.rstrip('\n')
    
    return x.rstrip('\n')

def getDataInString(filename):
    with open(filename) as f1:
        lst1 = map(rstripfn,f1.readlines())
    return lst1

def findngram(gramsize, lst1):
    lst1g = ngram(lst1, gramsize)

    reqdim = len(lst1g[0])
    lst1f = []
    for i in lst1g:
        lst1f.append(map(float, i))

    nplst1 = np.array(lst1f)
    return (nplst1,reqdim)

def preprocess_train(nplst1):
    lstf1 = nplst1.astype(float)
    min_max_scaler = preprocessing.MinMaxScaler()
    lstS1 = min_max_scaler.fit_transform(lstf1)
    return lstS1, min_max_scaler

def preprocess_test(nplst1, min_max_scaler):
    lstf1 = nplst1.astype(float)
    lstS1 = min_max_scaler.transform(lstf1)
    return lstS1

def trainfn(lstS1, n_compo, maxiter):
    clf = mixture.GaussianMixture(n_components=n_compo, covariance_type='full', max_iter=maxiter)
    clf.fit(lstS1)
    return clf

def findmeansigma(clf, lstS1):
    meanscore = np.mean(clf.score_samples(lstS1))
    meanvar = np.var(clf.score_samples(lstS1))
    sigma = math.sqrt(meanvar)
    return (meanscore, sigma)

def mypred(X, clf, mean, sigma):
    scores = clf.score_samples(X)
    predictions = []
    for i in scores:
        if(abs(i-mean)<thres*sigma):
            predictions.append(1)
        else:
            predictions.append(-1)
    return predictions

def calcncompo(n):
    if(n>10000):
        return 100
    elif(n<60):
        return max(1, n/10)
    else:
        return 9

def areAllNumeric(l):
    l1 = map(isCharacterAscii, l)
    return all(l1)

def isCharacterAscii(str):
    try:
        float(str)
        return True
    except ValueError:
        return False

def train(filename, sstableID):
    lst1 = getDataInString(filename)
    if(not areAllNumeric(lst1)):
        print("returning cause key is not numeric")
        return
    if(len(lst1) == 1):
        print('returning cause num key is 1')
        return
    nplst1, reqdim = findngram(gramsize, lst1)
    lstS1, min_max_scaler = preprocess_train(nplst1)
    n_compo = calcncompo(len(lst1))
    clf = trainfn(lstS1, n_compo, maxiter)
    mean, sigma = findmeansigma(clf, lstS1)
    models[str(sstableID)] = (clf, mean, sigma, reqdim, min_max_scaler)
    print('training done with sstableid - ', sstableID)
    return

def trainIndi(nplst1, sstableID,n_compo):
#     lst1 = getDataInString(filename)
#     if(not areAllNumeric(lst1)):
#         print("returning cause key is not numeric")
#         return
#     if(len(lst1) == 1):
#         print('returning cause num key is 1')
#         return
#     nplst1, reqdim = findngram(gramsize, lst1)
#     print("prev mean was ", np.mean(nplst1.transpose()))
#     print("prev var was ", np.var(nplst1.transpose()))
#     lstS1, min_max_scaler = preprocess_train(nplst1)
    print("passed mean was ",np.mean(nplst1.transpose()))
    print("passed sigma was ",np.var(nplst1.transpose()))
#     n_compo = calcncompo(len(lst1))
    clf = trainfn(nplst1, n_compo, maxiter)
    print("found means ")
    print(np.sort(clf.means_.transpose()[0]))
    print("cov matrix ")
    print(clf.covariances_)
    print("weights :")
    print(clf.weights_)
    mean, sigma = findmeansigma(clf, nplst1)
    print("mean is ",str(mean))
    print("sigma is ", str(sigma))
    reqdim = len(nplst1[0])
    models[str(sstableID)] = (clf, mean, sigma, reqdim)
    print('training done with sstableid - ', sstableID)
    return

def test(keyPredstr, sstableID):
    if(keyPredstr == ''):
        print('returning cause string is empty')
        return 0
    if(not isCharacterAscii(keyPredstr)):
        print('returning cause not numric')
        return 0
    keyPred = [str(keyPredstr)]
    if sstableID in models:
        # sizeofmodels()
        clf, mean, sigma, reqdim, min_max_scaler = models[str(sstableID)]
        keyPredg = ngramEnforce(keyPred, gramsize, reqdim)
        keyPredS = preprocess_test(keyPredg, min_max_scaler)
        prediction = mypred(keyPredS, clf, mean, sigma)
        return prediction[0]
    else:
        print(str(sstableID), ' - sstableid not in models')
        return 0

    
def testIndi(keyPredg, sstableID):
#     if(keyPredstr == ''):
#         print('returning cause string is empty')
#         return 0
#     if(not isCharacterAscii(keyPredstr)):
#         print('returning cause not numric')
#         return 0
#     keyPred = [str(keyPredstr)]
    if str(sstableID) in models:
        # sizeofmodels()
        clf, mean, sigma, reqdim= models[str(sstableID)]
#         keyPredg = ngramEnforce(keyPred, gramsize, reqdim)
#         keyPredS = preprocess_test([keyPredg], min_max_scaler)
        prediction = mypred([keyPredg], clf, mean, sigma)
        return prediction[0]
    else:
        print(str(sstableID), ' - sstableid not in models')
        return 0

def testIndi2(keyPredg, sstableID):
#     if(keyPredstr == ''):
#         print('returning cause string is empty')
#         return 0
#     if(not isCharacterAscii(keyPredstr)):
#         print('returning cause not numric')
#         return 0
#     keyPred = [str(keyPredstr)]
    if str(sstableID) in models:
        # sizeofmodels()
        clf, mean, sigma, reqdim= models[str(sstableID)]
#         keyPredg = ngramEnforce(keyPred, gramsize, reqdim)
#         keyPredS = preprocess_test([keyPredg], min_max_scaler)
        prediction = mypred(keyPredg, clf, mean, sigma)
        return prediction
    else:
        print(str(sstableID), ' - sstableid not in models')
        return 0
    
def sizeofmodels():
    l = []
    for sstableID in models:
        clf, mean, sigma, reqdim, min_max_scaler = models[str(sstableID)]
        print("yo")
        print(clf)
        p = pickle.dumps(clf)
        tempsize = sys.getsizeof(p)
        # print(str(tempsize))
        # print(sys.getsizeof(p))
        print(sstableID, sys.getsizeof(p))
def sumsizeofmodels(models):
    agg =0
    for sstableID in models:
        clf, mean, sigma, reqdim= models[str(sstableID)]
        print("yo")
        print(clf)
        p = pickle.dumps(clf)
        tempsize = sys.getsizeof(p)
        agg += tempsize
        # print(str(tempsize))
        # print(sys.getsizeof(p))
        print(sstableID, sys.getsizeof(p))
    return agg
def sumsizeofbloom(bloomfilters):
    agg =0
    for f in bloomfilters.values():
        agg += f.num_bits
    agg = agg/8
    return agg




# In[ ]:


gramsize = -1
maxiter = 100
thres = 1
models = {} # storing models based on sstable ids


# In[ ]:


f = BloomFilter(capacity=30, error_rate=0.1)


# In[ ]:


[f.add(0.01*x, skip_check=True) for x in range(100)]


# In[ ]:


all([(0.01*x in f) for x in range(100)])


# In[ ]:


10 in f


# In[ ]:


f.num_bits


# In[ ]:







# In[ ]:


# Generate the data

for i in range(0,numSstable):
    thistableData = generateGauss(numclustersPerSStable, gapBetweenClusters, clusterSigma, numelePerCluster )
    ssTableData[i] = thistableData


# In[ ]:


numclustersPerSStable = 100
numclustersPerSStableForTrain = numclustersPerSStable
numSstable = 30
numelePerCluster = 300
gapBetweenClusters = 0.5
clusterSigma = 0.1
ssTableData = {}
models = {}



# In[ ]:


thres = 0.1
bloomfilters = {}
capacityBloom = 30000
errorRate = 0.1


# In[ ]:


# Generate the data
alltemptableData = generateGaussArr(
    numclustersPerSStable*numSstable, gapBetweenClusters, clusterSigma, numelePerCluster )
np.random.shuffle(alltemptableData)
for i in range(0,numSstable):
    thistableData = alltemptableData[i*numclustersPerSStable:(i+1)*numclustersPerSStable].flatten()
    ssTableData[i] = thistableData.reshape(-1,1)


# In[ ]:


# ssTableData[0]


# In[ ]:


# alltemptableData


# In[ ]:


get_ipython().run_cell_magic(u'time', u'', u'# train the model on generated data\nfor i in range(0,numSstable):\n    thistableData = ssTableData[i]\n    trainIndi(thistableData,i, numclustersPerSStableForTrain)')


# In[ ]:


get_ipython().run_cell_magic(u'time', u'', u'# put into bloom filter false negatives\nfor i in range(0,numSstable):\n    thistableData = ssTableData[i]\n    thistableDataTranspose0 = thistableData.transpose()[0]\n    falseNegativeAns = testIndi2(thistableData, i)\n    numFalseNeg = falseNegativeAns.count(-1)\n#     newCapacity = int(capacityBloom*1.0*numFalseNeg/len(thistableData))\n    newCapacity = max(numFalseNeg,1)\n    print("newCapacity ",newCapacity)\n    f = BloomFilter(capacity=newCapacity, error_rate=errorRate)\n    for j in range(0,len(thistableDataTranspose0)):\n        if(falseNegativeAns[j]==-1):\n#             print("adding ", thistableDataTranspose0[j])\n            f.add(thistableDataTranspose0[j])\n    bloomfilters[i] = f')


# In[ ]:


get_ipython().run_cell_magic(u'time', u'', u'# create test data\n# For now test data is all data\n# calculate false positives for test data\ntruePositive=0\nfalsePositive=0\ntrueNegative=0\nfor i in range(0, numSstable):\n    print("on sstable number ", i)\n    thistableData = ssTableData[i]\n    thistableDataTranspose0 = thistableData.transpose()[0]\n    for elemind in range(0,len(thistableDataTranspose0),50):\n        elem = thistableDataTranspose0[elemind]\n        for j in range(0,numSstable):\n            finAnswer = (testIndi([elem],j)==1)\n            if not finAnswer:\n                finAnswer = elem in bloomfilters[j]\n            if finAnswer:\n                if(i==j):\n                    #True answer true positive\n                    #break because you found answer\n                    truePositive+=1\n                    break\n                else:\n                    #False answer false positive\n                    falsePositive+=1\n            else:\n                if(i==j):\n                    raise ValueEroor("not possible")\n                else:\n                    #True answer true negative\n                    trueNegative+=1\n                \n        ')


# In[ ]:


print(truePositive,falsePositive,trueNegative)


# In[ ]:


fprate = falsePositive*1.0/(falsePositive+trueNegative)
print(fprate)


# In[ ]:


sumsizeofmodels(models)


# In[ ]:


sumsizeofbloom(bloomfilters)


# In[ ]:


sys.getsizeof(models['0'])


# In[ ]:


def get_size(obj, seen=None):
    """Recursively finds size of objects"""
    size = sys.getsizeof(obj)
    if seen is None:
        seen = set()
    obj_id = id(obj)
    if obj_id in seen:
        return 0
    # Important mark as seen *before* entering recursion to gracefully handle
    # self-referential objects
    seen.add(obj_id)
    if isinstance(obj, dict):
        size += sum([get_size(v, seen) for v in obj.values()])
        size += sum([get_size(k, seen) for k in obj.keys()])
    elif hasattr(obj, '__dict__'):
        size += get_size(obj.__dict__, seen)
    elif hasattr(obj, '__iter__') and not isinstance(obj, (str, bytes, bytearray)):
        size += sum([get_size(i, seen) for i in obj])
    else:
        print(type(obj))
        print("cases left")
    return size


# In[ ]:


import sys
from types import ModuleType, FunctionType
from gc import get_referents

# Custom objects know their class.
# Function objects seem to know way too much, including modules.
# Exclude modules as well.
BLACKLIST = type, ModuleType, FunctionType


def getsize(obj):
    """sum size of object & members."""
    if isinstance(obj, BLACKLIST):
        raise TypeError('getsize() does not take argument of type: '+ str(type(obj)))
    seen_ids = set()
    size = 0
    objects = [obj]
    while objects:
        need_referents = []
        for obj in objects:
            if not isinstance(obj, BLACKLIST) and id(obj) not in seen_ids:
                seen_ids.add(id(obj))
                print(type(obj))
                print(sys.getsizeof(obj))
                size += sys.getsizeof(obj)
                print(size)
                need_referents.append(obj)
        objects = get_referents(*need_referents)
    return size


# In[ ]:


getsize(models['0'][0].weights_)


# In[ ]:


models['0'][0].covariances_.shape


# In[ ]:


type(models['0'][0].weights_[0])


# In[ ]:


tempint = 0
get_size(tempint)


# In[ ]:


inspect.getmembers(models['0'][0].means_, lambda a:not(inspect.isroutine(a)))


# In[ ]:


attributes = inspect.getmembers(models['0'][0], lambda a:not(inspect.isroutine(a)))


# In[ ]:


[a for a in attributes if not(a[0].startswith('__') and a[0].endswith('__'))]


# In[ ]:


mybool = True==False


# In[ ]:


ssTableData[0][:100]


# In[ ]:


testIndi2(ssTableData[0][:100],0)


# In[ ]:




