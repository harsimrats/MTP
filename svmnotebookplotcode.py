#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
get_ipython().magic(u'matplotlib inline')
from pybloom import BloomFilter
import inspect
import matplotlib.font_manager
from sklearn import svm
import os
import glob
import pickle as pkl
import time
get_ipython().magic(u'matplotlib notebook')
from itertools import product
from joblib import Parallel, delayed
import multiprocessing
from tqdm import tqdm


# In[2]:


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

def ngramMaxDimKnown(Xd, gramsize, maxlen):
    X = map(str, Xd)
    numele = len(X)
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
def ngramEnforceMaxKnown(Xd, gramsize, reqdim):
    X = map(str, Xd)
    numele = len(X)
    maxlen = reqdim + gramsize -1
    Y = []
    for i in X:
        tempy = []
        paddedi = i + "0"*(maxlen-len(i))
        for j in range(0,len(paddedi)-gramsize+1):
            tempy.append(paddedi[j:j+gramsize])
        templeny = len(tempy)
        
        Y.append(tempy)
    return Y

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
    print(scores)
    predictions = []
    for i in scores:
        print("diff is ",abs(i-mean))
        if(abs(i-mean)<=max(thres*sigma, 1.0/1e12)):
            predictions.append(1)
        else:
            predictions.append(-1)
    return predictions
def mypredSvm(X, clf):
    scores = clf.predict(X)
    return clf.predict(X)

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

def trainIndiSvm(nplst1, sstableID, nu, gamma, models):
    print("passed mean was ",np.mean(nplst1, axis=0))
    print("passed sigma was ",np.var(nplst1, axis=0))
    clf = svm.OneClassSVM(nu=nu, kernel="rbf", gamma=gamma)
    clf.fit(nplst1)
    reqdim = len(nplst1[0])
#     print("reqdim is ",reqdim)
    models[str(sstableID)] = (clf, reqdim)
    print('training done with sstableid - ', sstableID)
    return models
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
def testIndiSvm(keyPredg, sstableID, models):
#     if(keyPredstr == ''):
#         print('returning cause string is empty')
#         return 0
#     if(not isCharacterAscii(keyPredstr)):
#         print('returning cause not numric')
#         return 0
#     keyPred = [str(keyPredstr)]
    if str(sstableID) in models:
        # sizeofmodels()
        clf, reqdim= models[str(sstableID)]
#         keyPredg = ngramEnforce(keyPred, gramsize, reqdim)
#         keyPredS = preprocess_test([keyPredg], min_max_scaler)
        prediction = mypredSvm([keyPredg], clf)
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
def testIndiSvm2(keyPredg, sstableID, models):
#     if(keyPredstr == ''):
#         print('returning cause string is empty')
#         return 0
#     if(not isCharacterAscii(keyPredstr)):
#         print('returning cause not numric')
#         return 0
#     keyPred = [str(keyPredstr)]
    if str(sstableID) in models:
        # sizeofmodels()
        clf, reqdim= models[str(sstableID)]
#         keyPredg = ngramEnforce(keyPred, gramsize, reqdim)
#         keyPredS = preprocess_test([keyPredg], min_max_scaler)
        prediction = mypredSvm(keyPredg, clf)
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
def sumsizeofmodelssvm(models):
    agg =0
    for sstableID in models:
        clf, reqdim= models[str(sstableID)]
        print("yo")
        print(clf)
        p = pickle.dumps(clf)
        tempsize = sys.getsizeof(p)
        agg += tempsize
        # print(str(tempsize))
        # print(sys.getsizeof(p))
        print(sstableID, sys.getsizeof(p))
    return agg
def sumsizeofmodelssvmReal(models):
    agg =0
    for sstableID in models:
        tempmodel, reqdim= models[str(sstableID)]
        tempsize = 8*(tempmodel.support_vectors_.shape[0]*tempmodel.support_vectors_.shape[1])+ 8*(tempmodel.dual_coef_.shape[0]*tempmodel.dual_coef_.shape[1]) + 4*(tempmodel.support_.shape[0])
        agg += tempsize
        # print(str(tempsize))
        # print(sys.getsizeof(p))
        print(sstableID, tempsize)
    return agg
def sumsizeofbloom(bloomfilters):
    agg =0
    for f in bloomfilters.values():
        agg += f.num_bits
    agg = agg/8
    return agg

def readFileByLines(filename):
    f = open(filename, 'r')
    lst = f.readlines()
    f.close()
#     print(lst)
#     lst = map(str.rstrip, lst)
    return lst
def parseVdisk(lst):
    lst = [s.split(':') for s in lst]
#     vid = []
#     blc = [] 
    mat = []
    for ele in lst:
#         vid.append(int(ele[1]))
#         blc.append(int(ele[2], 16))
        mat.append([float(int(ele[1]))/1e0,float(int(ele[2], 16))/1e0])
#     print(vid, blc)
#     return (vid, blc)
    return mat
def createNgramsPerSStable(thisssTableData, alltemptableData):
    mingram = 4
    maxgram = 4
    initialDimension = alltemptableData.shape[1]
    
    extrapolatedData = thisssTableData
    findimlist = []
    for realFeature in range(initialDimension):
    #     not taking care of negative features
        thisFeatureData = thisssTableData[:,realFeature]
        maxlen = len(str(np.max(alltemptableData[:,realFeature])))
        dimlist = []
        for gramsize in range(mingram, min(maxgram,maxlen)+1):
#             print(thisFeatureData[0], gramsize, maxlen)
#             print(gramsize)
            thisFeatureDataOfGramSize = np.array(ngramMaxDimKnown(thisFeatureData, gramsize, maxlen),dtype=np.float)
#             print(thisFeatureDataOfGramSize[0])
#             print(thisFeatureDataOfGramSize.shape)
            reqdimForLater = thisFeatureDataOfGramSize.shape[1]
#             print(reqdimForLater)
            dimlist.append(reqdimForLater)
            extrapolatedData = np.append(extrapolatedData, thisFeatureDataOfGramSize, axis=1)
        findimlist += dimlist
    return (extrapolatedData, findimlist)
def createNgrams(ssTableData, alltemptableData):
    ngramData = {}
    dimlists = []
    for i in range(len(ssTableData)):
        (bigData, dimlist)=createNgramsPerSStable(ssTableData[i], alltemptableData)
        ngramData[i] = bigData
        dimlists.append(dimlist)
    return (ngramData, dimlists)
# def createNgramsPerSStableEnforce(thisssTableData, alltemptableData, dimlist):
#     mingram = 3
#     maxgram = 7
#     initialDimension = alltemptableData.shape[1]
    
#     extrapolatedData = thisssTableData
#     findimlist = []
#     for realFeature in range(initialDimension):
#     #     not taking care of negative features
#         thisFeatureData = thisssTableData[:,realFeature]
#         maxlen = len(str(np.max(alltemptableData[:,realFeature])))
#         dimlist = []
#         for gramsize in range(mingram, min(maxgram,maxlen)+1):
# #             print(thisFeatureData[0], gramsize, maxlen)
# #             print(gramsize)
#             thisFeatureDataOfGramSize = np.array(ngramMaxDimKnown(thisFeatureData, gramsize, maxlen),dtype=np.float)
# #             print(thisFeatureDataOfGramSize[0])
# #             print(thisFeatureDataOfGramSize.shape)
#             reqdimForLater = thisFeatureDataOfGramSize.shape[1]
# #             print(reqdimForLater)
#             dimlist.append(reqdimForLater)
#             extrapolatedData = np.append(extrapolatedData, thisFeatureDataOfGramSize, axis=1)
#         findimlist += dimlist
#     return (extrapolatedData, findimlist)


# In[3]:


def getData():
    foldername = "metadata_maps/vdisk/*.db"
    files = sorted(glob.glob(foldername))
    ctr=0
    alltemptableData = []
    for fle in files:
        thisline = readFileByLines(fle)
    #     if(len(thisline)>1000):
    #         continue
        mat = parseVdisk(thisline)
        ssTableData[ctr] = np.array(mat).reshape(-1,2)
        ctr+=1
        alltemptableData+=mat
        print(ctr,len(thisline))
    # print(lines)
    alltemptableData = np.array(alltemptableData).reshape(-1,2)
    return (ssTableData, alltemptableData)


# In[4]:


def getScalers():
    scalers = {}
    for i in range(0,len(ssTableData)):
        thistableData = ssTableData[i]
        scaler = preprocessing.StandardScaler().fit(thistableData)
        scalers[i] = scaler
    return scalers


# In[5]:


# Generate the data
# alltemptableData = generateGaussArrTwoDim(
#     numclustersPerSStable*numSstable, gapBetweenClusters, clusterSigma, numelePerCluster )
# np.random.shuffle(alltemptableData)


# In[6]:


# for i in range(0,len(ssTableData)):
#     thistableData = alltemptableData[i*numclustersPerSStable:(i+1)*numclustersPerSStable]
#     ssTableData[i] = thistableData.reshape(-1,thistableData.shape[2])


# In[7]:


# %%time
# thistableData = ssTableData[0]
# trainIndiSvm(scalers[0].transform(thistableData),0, nu, gamma)


# In[ ]:





# In[8]:


# %%time
def trainFn(nu, gamma):
    # train the model on generated data
    models = {}
    for i in range(0,len(ssTableData)):
        starttime = time.time()
        thistableData = ssTableData[i]
        models = trainIndiSvm(scalers[i].transform(thistableData),i, nu, gamma, models)
        endtime = time.time()
        print(str(nu),str(gamma),"time taken ", (endtime-starttime))
    return models
def trainFnTol(nu, gamma, tol):
    # train the model on generated data
    models = {}
    for i in range(0,len(ssTableData)):
        starttime = time.time()
        thistableData = ssTableData[i]
        models = trainIndiSvmTol(scalers[i].transform(thistableData),i, nu, gamma, models, tol)
        endtime = time.time()
        print(str(nu),str(gamma),"time taken ", (endtime-starttime))
    return models


# In[9]:


# pklfile = open("svmmodelstandardscaled_0.1_50", "rb")
# models = pkl.load(pklfile)
# pklfile.close()


# In[10]:


# %%time
# put into bloom filter false negatives
def putIntoBloomFilters(models):
    bloomfilters = {}
    errorsTrain = {}
    for i in range(0,len(ssTableData)):
        thistableData = ssTableData[i]
        thistableDataTranspose0 = thistableData.transpose()[0]
        if removeClassifier:
            newCapacity = max(len(thistableData),1)
            print("newCapacity ",newCapacity)
            f = BloomFilter(capacity=newCapacity, error_rate=errorRate)
            for j in range(0,len(thistableData)):
                f.add(separator.join(map(str,thistableData[j])))
            bloomfilters[i] = f
        else:
            falseNegativeAns = testIndiSvm2(scalers[i].transform(thistableData), i, models)
        #     print(falseNegativeAns)
            numFalseNeg = falseNegativeAns.tolist().count(-1)
        #     newCapacity = int(capacityBloom*1.0*numFalseNeg/len(thistableData))
            newCapacity = max(numFalseNeg,1)
            print("newCapacity ",newCapacity)
            f = BloomFilter(capacity=newCapacity, error_rate=errorRate)
            for j in range(0,len(thistableData)):
                if(falseNegativeAns[j]==-1 or removeClassifier):
        #             print("adding ", thistableDataTranspose0[j])
                    f.add(separator.join(map(str,thistableData[j])))
            bloomfilters[i] = f
        errorsTrain[i] = newCapacity
    return (bloomfilters,errorsTrain)


# In[11]:


# %%time
def performTests(models, bloomfilters):
    # create test data
    # For now test data is all data
    # calculate false positives for test data
    truePositive_c=0
    truePositive_bf=0
    truePositive=0
    falsePositive_c=0
    falsePositive_bf=0
    falsePositive=0
    trueNegative=0
    for i in range(0, len(ssTableData)):
        print("on sstable number ", i)
        thistableData = ssTableData[i]
        thistableDataTranspose0 = thistableData.transpose()[0]
        for elemind in range(0,len(thistableData),skipFactor):
            elem = thistableData[elemind]
            for j in range(0,len(ssTableData)):
                if removeClassifier:
                    cAnswer = False
                else:
                    cAnswer = (testIndiSvm(scalers[j].transform([elem])[0],j, models)==1)
    #             print(cAnswer)
                if not cAnswer:
                    bfAnswer = separator.join(map(str,elem)) in bloomfilters[j]
                    finAnswer = bfAnswer
                else:
                    bfAnswer = None
                    finAnswer = cAnswer
                if finAnswer:
                    if(i==j):
                        #True answer true positive
                        #break because you found answer
                        truePositive+=1
                        if bfAnswer is None:
                            truePositive_c+=1
                        else:
                            truePositive_bf+=1
                        break
                    else:
                        #False answer false positive
                        falsePositive+=1
                        if bfAnswer is None:
                            falsePositive_c+=1
                        else:
                            falsePositive_bf+=1
                else:
                    if(i==j):
                        raise ValueEroor("not possible")
                    else:
                        #True answer true negative
                        trueNegative+=1
    fprate = falsePositive*1.0/(falsePositive+trueNegative)
    return (fprate,
            truePositive_c,truePositive_bf,truePositive,
            falsePositive_c,falsePositive_bf,falsePositive,
            trueNegative)
                
        


# In[12]:


def persistModel(models, nu, gamma):
    pickle_out = open("svmModels/svmmodelstandardscaled_"+namePrefix+str(nu)+"_"+str(gamma),"wb")
    pkl.dump(models, pickle_out)
    pickle_out.close()


# In[13]:


def loadModel(nu, gamma):
    pickle_out = open("svmModels/svmmodelstandardscaled_"+namePrefixOri+str(nu)+"_"+str(gamma),"rb")
    models = pkl.load(pickle_out)
    pickle_out.close()
    return models


# In[14]:


def persistBloomFilters(bloomfilters, nu, gamma):
    pickle_out = open("bloomfilters/bloomfilters_"+namePrefix+str(nu)+"_"+str(gamma),"wb")
    pkl.dump(bloomfilters, pickle_out)
    pickle_out.close()


# In[15]:


def loadBloomFilters(nu, gamma):
    pickle_out = open("bloomfilters/bloomfilters_"+namePrefix+str(nu)+"_"+str(gamma),"rb")
    bloomfilters = pkl.load(pickle_out)
    pickle_out.close()
    return bloomfilters


# In[16]:


def persistObservations(obs, nu, gamma):
    pickle_out = open("Observations/observations_"+namePrefix+str(nu)+"_"+str(gamma),"wb")
    pkl.dump(obs, pickle_out)
    pickle_out.close()


# In[17]:


def loadObservations(nu, gamma):
    pickle_out = open("Observations/observations_"+namePrefix+str(nu)+"_"+str(gamma),"rb")
    obs = pkl.load(pickle_out)
    pickle_out.close()
    return obs


# In[18]:


def persistTimes(obs):
    pickle_out = open("times/times_"+namePrefix,"wb")
    pkl.dump(obs, pickle_out)
    pickle_out.close()


# In[19]:


def loadTimes():
    pickle_out = open("times/times_"+namePrefix,"rb")
    obs = pkl.load(pickle_out)
    pickle_out.close()
    return obs


# In[20]:


def ensemble(nu, gamma):
    print(nu,gamma)
    starttime = time.time()
    models = trainFn(nu, gamma)
    persistModel(models, nu, gamma)
#     models = loadModel(nu, gamma)
    (bloomfilters, errorTrain) = putIntoBloomFilters(models)
    persistBloomFilters(bloomfilters, nu, gamma)
    (fprate,
    truePositive_c,truePositive_bf,truePositive,
    falsePositive_c,falsePositive_bf,falsePositive,
    trueNegative) = performTests(models, bloomfilters)
    observations = (errorTrain, fprate,
                    truePositive_c,truePositive_bf,truePositive,
                    falsePositive_c,falsePositive_bf,falsePositive,
                    trueNegative)
    persistObservations(observations, nu, gamma)
    endtime = time.time()
    return (endtime-starttime, nu, gamma)


# In[21]:


gramsize = -1
maxiter = 100
# models = {} # storing models based on sstable ids
# nu = 0.1
# gamma = 1
removeClassifier = False


# In[22]:


# thres = -0.1
# bloomfilters = {}
capacityBloom = 30000
errorRate = 0.1
skipFactor = 50


# In[23]:


ssTableData = {}
separator = ":"
namePrefix = "6nu6gamma_ex4_e2"
namePrefixOri = "6nu6gamma_ex4"


# In[24]:


get_ipython().run_cell_magic(u'time', u'', u'(ssTableData, alltemptableData) = getData()\noriSsTableData = ssTableData.copy()\nssTableData, dimlist = createNgrams(ssTableData, alltemptableData)\nscalers = getScalers()\n# nus = np.logspace(-3,0,6)\n# gammas = np.logspace(-3,3,6)\nnus = np.logspace(-3,0,6)\ngammas = np.logspace(-1,3,6)\nnum_cores = multiprocessing.cpu_count()\n# (ssTableData, alltemptableData) = getData()\n# scalers = getScalers()\n# nus = np.logspace(-6,0,20)\n# gammas = np.logspace(-6,3,20)\n# # nus = np.logspace(-1,0,1)\n# # gammas = np.logspace(0,3,1)\n# num_cores = multiprocessing.cpu_count()\n\n# results = Parallel(n_jobs=num_cores)(delayed(ensemble)(i,j) for i,j in tqdm(product(nus, gammas)))\n# print(results)\n# # for nu, gamma in product(nus, gammas):\n# #     ensemble(nu,gamma)\n    \n# # models = trainFn(0.1,1)')


# In[25]:


ssTableData


# In[26]:


mshape = (6,6)
# mshape = (1,1)


# In[27]:


# nus = np.logspace(-6,0,mshape[0])
# gammas = np.logspace(-6,3,mshape[1])
# nus = np.logspace(-3,0,1)
# gammas = np.logspace(1,3,1)


# In[28]:


results = loadTimes()


# In[29]:


# namePrefix = "6nu6gamma_ex"


# In[30]:


results


# In[31]:


nus


# In[32]:


gammas


# In[33]:


obsdic = {}
fprates = np.zeros(mshape)
fprateslist = []
bfsizelist = []
modelSizeList = []
fpdic = {}
for i,j in product(nus,gammas):
    print(i,j)
    try:
        obs = loadObservations(i,j)
        fprateslist.append(obs[1])
        fpdic[(i,j)] = obs[1]
        print(obs[1])
    except IOError:
        obs = None
        fprateslist.append(1)
        fpdic[(i,j)] = None
        print(obs)
    try:
        bloomfilters = loadBloomFilters(i,j)
        tempsize = sumsizeofbloom(bloomfilters)
        bfsizelist.append(tempsize)
        print(tempsize)
    except IOError:
        tempsize = None
        bfsizelist.append(1e6)
        print(tempsize)
    try:
        models = loadModel(i,j)
        tempsize = sumsizeofmodelssvmReal(models)
        modelSizeList.append(tempsize)
        print(tempsize)
    except IOError:
        tempsize = None
        modelSizeList.append(1e10)
        print(tempsize)
    
#     fprateslist.append(obs[1])
#     fpdic[(i,j)] = obs[1]
    obsdic[(i,j)] = obs
fprateslist = np.array(fprateslist).reshape(mshape)
bfsizelist = np.array(bfsizelist).reshape(mshape)
modelSizeList = np.array(modelSizeList).reshape(mshape)
totalSizeList = bfsizelist + modelSizeList


# In[34]:


obsdic


# In[35]:


minval = 1.0
for i in obsdic:
    print(i, obsdic[i][1])
    if obsdic[i][1] < minval:
        minval = obsdic[i][1]
        minpair =i


# In[36]:


minval


# In[37]:


loadTimes()


# In[38]:


minpair


# In[39]:


modelSizeList


# In[40]:


import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_iris
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import GridSearchCV


# Utility function to move the midpoint of a colormap to be around
# the values of interest.

class MidpointNormalize(Normalize):

    def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
        self.midpoint = midpoint
        Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
        return np.ma.masked_array(np.interp(value, x, y))
plt.figure(figsize=(8,5))
plt.subplots_adjust(left=.2, right=0.95, bottom=0.15, top=0.95)
plt.imshow(fprateslist, interpolation='nearest', cmap=plt.cm.hot,
           norm=MidpointNormalize(vmin=0.2, midpoint=0.92))
plt.xlabel('gamma')
plt.ylabel('C')
plt.colorbar()
plt.yticks(np.arange(len(nus)), nus, rotation=45)
plt.xticks(np.arange(len(gammas)), gammas)
plt.title('Validation accuracy')
plt.show()


# In[41]:


import seaborn as sns; sns.set(rc={'figure.figsize':(30,20)})
ax = sns.heatmap(bfsizelist, xticklabels=gammas, yticklabels = nus, cmap="YlGnBu", vmin=0, annot=True)


# In[42]:


import seaborn as sns; sns.set(rc={'figure.figsize':(30,20)})
ax = sns.heatmap(modelSizeList, xticklabels=gammas, yticklabels = nus, cmap="YlGnBu", vmin=0, annot=True)


# In[43]:


import seaborn as sns; sns.set(rc={'figure.figsize':(30,20)})
ax = sns.heatmap(fprateslist, xticklabels=gammas, yticklabels = nus, cmap="YlGnBu", vmin=0,vmax=1, annot=True)


# In[44]:


import seaborn as sns; sns.set(rc={'figure.figsize':(30,20)})
ax = sns.heatmap(totalSizeList, xticklabels=gammas, yticklabels = nus, cmap="YlGnBu", vmin=0, annot=True)


# In[45]:


(fprateslist * bfsizelist).shape


# In[46]:


bestval = (fprateslist[0][0],totalSizeList[0][0])
bestpair = (0,0)
for i in range(6):
    for j in range(6):
        thisfpr = fprateslist[i][j]
        thismsize = totalSizeList[i][j]
        if(thisfpr<0.2 and thismsize<bestval[1]):
            bestval = (thisfpr,thismsize)
            bestpair = (i,j)
print(bestpair)
print(bestval) 
print(sumsizeofbloom(coreBf))
print(bestval[1]*1.0/sumsizeofbloom(coreBf)*1.0)
print(bestval[0]*1.0/coreobs[1])


# In[ ]:


bestval = fprateslist[0][0]
bestpair = (0,0)
for i in range(20):
    for j in range(20):
        thisfpr = fprateslist[i][j]
        thismsize = totalSizeList[i][j]
        if(thisfpr<coreobs[1]):
            bestval = thisfpr
            bestpair = (i,j)
print(bestpair)
print(bestval) 
print(sumsizeofbloom(coreBf))
print(bestval*1.0/sumsizeofbloom(coreBf)*1.0)
print(bestval*1.0/coreobs[1])


# In[ ]:


coreobs[1]


# In[47]:


import seaborn as sns; sns.set(rc={'figure.figsize':(30,20)})
ax = sns.heatmap((np.power(fprateslist,3) * totalSizeList), xticklabels=gammas, yticklabels = nus, cmap="YlGnBu", annot=True)


# In[48]:


import seaborn as sns; sns.set(rc={'figure.figsize':(30,20)})
ax = sns.heatmap(modelSizeList, xticklabels=gammas, yticklabels = nus, cmap="YlGnBu", vmin=0, annot=True)


# In[49]:


import seaborn as sns; sns.set(rc={'figure.figsize':(30,20)})
ax = sns.heatmap(modelSizeList+bfsizelist, xticklabels=gammas, yticklabels = nus, cmap="YlGnBu", vmin=0,vmax = 1e6, annot=True)


# In[ ]:


modelSizeList + bfsizelist*0


# In[ ]:





# In[ ]:


fpdic


# In[ ]:


namePrefix = "core"
coreobs = loadObservations(0.1,1.0)
coreBf = loadBloomFilters(0.1,1.0)
print(sumsizeofbloom(coreBf))


# In[ ]:


coreobs


# In[ ]:


nunumber = 16
gammanumber = 18
tempmodel = loadModel(nus[nunumber-1],gammas[gammanumber-1])
obs = loadObservations(nus[nunumber-1],gammas[gammanumber-1])
print(nus[nunumber-1],gammas[gammanumber-1])


# In[ ]:


namePrefix = '20nu20gamma'


# In[ ]:


tempmodel.fit_status_


# In[ ]:


tempmodel.offset_


# In[ ]:


len(tempmodel.support_)


# In[ ]:


tempmodel.support_


# In[ ]:


tempmodel.support_vectors_


# In[ ]:


len(set(tempmodel.dual_coef_[0]))


# In[ ]:


len(tempmodel.dual_coef_[0])


# In[ ]:


tempmodel.dual_coef_[0].tolist().count(1.0)


# In[ ]:


get_ipython().magic(u'pinfo tempmodel')


# In[ ]:


tempmodel.intercept_


# In[ ]:


type(tempmodel.support_)


# In[ ]:


tempmodel.support_.dtype


# In[ ]:


tempmodel.support_vectors_.dtype


# In[ ]:


tempmodel.dual_coef_.dtype


# In[ ]:


tempmodel.intercept_


# In[ ]:


8*(tempmodel.support_vectors_.shape[0]*tempmodel.support_vectors_.shape[1])+ 8*(tempmodel.dual_coef_.shape[0]*tempmodel.dual_coef_.shape[1]) + 4*(tempmodel.support_.shape[0])


# In[ ]:


tempmodel.support_.shape[0]


# In[ ]:


sumsizeofmodelssvmReal(models)


# In[ ]:


# nunumber = 18
# gammanumber = 18
# # tempmodel = loadModel(nus[nunumber-1],gammas[gammanumber-1])
# tempmodel = trainFnTol(nus[nunumber-1],gammas[gammanumber-1],1e-5)
# print(nus[nunumber-1],gammas[gammanumber-1])


# In[ ]:


# namePrefix = namePrefix + "tol1e-5"
# persistModel(tempmodel,nus[nunumber-1],gammas[gammanumber-1])


# In[ ]:


ssTableNumber = 2
temptableData = ssTableData[ssTableNumber]
clf = tempmodel[str(ssTableNumber)][0]
xx, yy = np.meshgrid(
    np.linspace(
        min(scalers[ssTableNumber].transform(temptableData.reshape(-1,2)).T[0]), 
        max(scalers[ssTableNumber].transform(temptableData.reshape(-1,2)).T[0]), 500), 
    np.linspace(
        min(scalers[ssTableNumber].transform(temptableData.reshape(-1,2)).T[1]), 
        max(scalers[ssTableNumber].transform(temptableData.reshape(-1,2)).T[1]), 500))
Z = clf.decision_function(np.c_[xx.ravel(),yy.ravel()])
Z = Z.reshape(xx.shape)
thisScaleddata = scalers[ssTableNumber].transform(temptableData)
thistablepredicitons = clf.predict(thisScaleddata)



# In[ ]:


testData = []
for i in range(len(ssTableData)):
    if i==ssTableNumber:
        continue
    thistabledata = ssTableData[i]
    for j in range(0, len(thistabledata), skipFactor):
        testData.append(thistabledata[j])
testData = np.array(testData)


# In[ ]:


testScaleddata = scalers[ssTableNumber].transform(testData)
testtablepredicitons = clf.predict(testScaleddata)


# In[ ]:


errorTestCases = np.array([testScaleddata[i] for i in range(len(testScaleddata)) if testtablepredicitons[i]==1])


# In[ ]:


errorTrainCases = np.array([thisScaleddata[i] for i in range(len(thisScaleddata)) if thistablepredicitons[i]==-1 ])


# In[ ]:



plt.title("Novelty Detection")
plt.contourf(xx, yy, Z, levels=np.linspace(Z.min(), 0, 7), cmap=plt.cm.PuBu)
a = plt.contour(xx, yy, Z, levels=[0], linewidths=2, colors='darkred')
plt.contourf(xx, yy, Z, levels=[0, Z.max()], colors='palevioletred')
s = 40

# testData = np.array([ssTableData[i] for i in ssTableData if i!=ssTableNumber]).reshape(-1,2)
b2 = plt.scatter(errorTestCases[:, 0], errorTestCases[:, 1], c='blueviolet', s=s, edgecolors='k')
# b1 = plt.scatter(
#     scalers[ssTableNumber].transform(ssTableData[ssTableNumber])[:, 0], 
#     scalers[ssTableNumber].transform(ssTableData[ssTableNumber])[:, 1], c='white', s=s, edgecolors='k')
# b3 = plt.scatter(errorTrainCases[:,0], errorTrainCases[:,1], c='yellow', s=s, edgecolors='k')
# b4 = plt.scatter(clf.support_vectors_[:,0], clf.support_vectors_[:,1], c= 'green', s=s, edgecolors='k')
# c = plt.scatter(X_outliers[:, 0], X_outliers[:, 0]*0+2, c='gold', s=s,
#                 edgecolors='k')
plt.axis('tight')
# plt.xlim((-5, 5))
# plt.ylim((-5, 5))
# plt.show()
plt.legend([a.collections[0], b2],
           ["learned frontier", "error test observations"],
           prop=matplotlib.font_manager.FontProperties(size=11))
plt.xlabel(
    "error train: %d/%d ; errors test: %d/%d ; false positives: %d ; fpr: %f"
    % (len(errorTrainCases), len(thisScaleddata), len(errorTestCases), len(testScaleddata), obs[7], obs[1]))
plt.show()


# In[ ]:


plt.savefig("plots/"+namePrefix + str(nus[nunumber-1])+"_"+str(gammas[gammanumber-1]) +"_"+ str(ssTableNumber) +"_"+ str(nunumber)+"_"+str(gammanumber)+".png")


# In[ ]:





# In[ ]:



errorTrainCases =  np.array(errorTrainCases)


# In[ ]:


map(scalers[0].transform, ssTableData[0])


# In[ ]:


thisScaleddata = scalers[ssTableNumber].transform(temptableData)
thistablepredicitons = clf.predict(thisScaleddata)
[thisScaleddata[i] for i in range(len(thisScaleddata)) if thistablepredicitons[i]==-1]


# In[ ]:


thistablepredicitons.tolist().count(-1)


# In[ ]:


thistablepredicitons.tolist().count(1)


# In[ ]:


np.linspace(Z.min(), 0, 7)


# In[ ]:


Z.max()


# In[ ]:


clf.get_params()


# In[ ]:


tempscores = clf.score_samples(thisScaleddata)


# In[ ]:


wrongscores = [tempscores[i] for i in range(len(thisScaleddata)) if thistablepredicitons[i]==-1]


# In[ ]:


min(wrongscores)


# In[ ]:


max(wrongscores)


# In[ ]:


tempscores = tempscores.tolist()


# In[ ]:


tempscores.sort(reverse=True)


# In[ ]:


tempscores


# In[ ]:


clf.intercept_


# In[ ]:


np.sort(wrongscores + clf.intercept_[0])


# In[ ]:


plt.hist(np.sort(wrongscores + clf.intercept_[0]), normed=True, bins=40)
plt.ylabel('Probability');


# In[ ]:


np.sort(tempscores)


# In[ ]:


tempscores


# In[ ]:


errorTrainCases


# In[ ]:


wrongscores


# In[ ]:


clf.support_vectors_


# In[ ]:


scoressup = clf.score_samples(clf.support_vectors_)
predictionssup = clf.predict(clf.support_vectors_)


# In[ ]:


np.sort(scoressup+clf.intercept_[0])[-15000:]


# In[ ]:


plt.hist(np.sort(scoressup+clf.intercept_[0]), normed=True, bins=400)
plt.ylabel('Probability');


# In[ ]:


clf.tol


# In[ ]:


len(scoressup)


# In[ ]:




