import sys
import numpy as np
import matplotlib.pyplot as plt
# get_ipython().magic(u'matplotlib inline')
from pybloom import BloomFilter
import inspect
import matplotlib.font_manager
from sklearn import svm
import os
import glob
import pickle as pkl
import time
# get_ipython().magic(u'matplotlib notebook')
from itertools import product
from joblib import Parallel, delayed
import multiprocessing
from tqdm import tqdm
from sklearn import preprocessing
from sklearn.metrics import accuracy_score
import pickle
from sklearn import mixture
import math
import epsilon_occ as ep

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



# In[12]:


# def persistModel(models, D, epsilon, agg_thr, box_thr):
#     pickle_out = open("svmModels/svmmodelstandardscaled_"+namePrefix+str(D)+"_"+str(epsilon)+"_"+str(agg_thr)+"_"+str(box_thr),"wb")
#     pkl.dump(models, pickle_out)
#     pickle_out.close()

def persistModel(models, D, epsilon, agg_thr, box_thr):
    pickle_out = open("svmModels/svmmodelstandardscaled_"+namePrefix+str(D)+"_"+str(epsilon)+"_"+str(agg_thr)+"_"+str(box_thr),"wb")
    pkl.dump(models, pickle_out)
    pickle_out.close()


# In[13]:


def loadModel(D, epsilon, agg_thr, box_thr):
    pickle_out = open("svmModels/svmmodelstandardscaled_"+namePrefixOri+str(D)+"_"+str(epsilon)+"_"+str(agg_thr)+"_"+str(box_thr),"rb")
    models = pkl.load(pickle_out)
    pickle_out.close()
    return models


# In[14]:


def persistBloomFilters(bloomfilters, D, epsilon, agg_thr, box_thr):
    pickle_out = open("bloomfilters/bloomfilters_"+namePrefix+str(D)+"_"+str(epsilon)+"_"+str(agg_thr)+"_"+str(box_thr),"wb")
    pkl.dump(bloomfilters, pickle_out)
    pickle_out.close()


# In[15]:


def persistObservations(obs, D, epsilon, agg_thr, box_thr):
    print(obs, D, epsilon, agg_thr, box_thr)
    pickle_out = open("Observations/observations_"+namePrefix+str(D)+"_"+str(epsilon)+"_"+str(agg_thr)+"_"+str(box_thr),"wb")
    pkl.dump(obs, pickle_out)
    pickle_out.close()


# In[16]:


def persistTimes(obs):
    pickle_out = open("times/times_"+namePrefix,"wb")
    pkl.dump(obs, pickle_out)
    pickle_out.close()

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

def getScalers():
    scalers = {}
    for i in range(0,len(ssTableData)):
        thistableData = ssTableData[i]
        scaler = preprocessing.StandardScaler().fit(thistableData)
        scalers[i] = scaler
    return scalers

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

def createNgramsPerSStable(thisssTableData, alltemptableData, mingram, maxgram):
    # mingram = 6
    # maxgram = 6
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
def createNgrams(ssTableData, alltemptableData, mingram, maxgram):
    ngramData = {}
    dimlists = []
    for i in range(len(ssTableData)):
        (bigData, dimlist)=createNgramsPerSStable(ssTableData[i], alltemptableData, mingram, maxgram)
        ngramData[i] = bigData
        dimlists.append(dimlist)
    return (ngramData, dimlists)

# def trainIndiSvm(nplst1, sstableID, D, epsilon, agg_thr, box_thr, models):
#     print("passed mean was ",np.mean(nplst1, axis=0))
#     print("passed sigma was ",np.var(nplst1, axis=0))
#     clf = svm.OneClassSVM(nu=nu, kernel="rbf", gamma=gamma)
#     clf.fit(nplst1)
#     reqdim = len(nplst1[0])
# #     print("reqdim is ",reqdim)
#     models[str(sstableID)] = (clf, reqdim)
#     print('training done with sstableid - ', sstableID)
#     return models

def trainIndiEpsilon(nplst1, sstableID, D, epsilon, agg_thr, box_thr, models):
    print("passed mean was ",np.mean(nplst1, axis=0))
    print("passed sigma was ",np.var(nplst1, axis=0))
    clf = ep.RPEOCC(D, epsilon, agg_thr, box_thr)
    clf.fit(nplst1)
    reqdim = len(nplst1[0])
#     print("reqdim is ",reqdim)
    models[str(sstableID)] = (clf, reqdim)
    print('training done with sstableid - ', sstableID)
    return models

# def trainFn(D, epsilon, agg_thr, box_thr):
#     # train the model on generated data
#     models = {}
#     for i in range(0,len(ssTableData)):
#         starttime = time.time()
#         thistableData = ssTableData[i]
#         models = trainIndiSvm(scalers[i].transform(thistableData),i, D, epsilon, agg_thr, box_thr, models)
#         endtime = time.time()
#         print(str(nu),str(gamma),"time taken ", (endtime-starttime))
#     return models
def trainEpsilon(D, epsilon, agg_thr, box_thr):
    models = {}
    for i in range(0,len(ssTableData)):
        starttime = time.time()
        thistableData = ssTableData[i]
        models = trainIndiEpsilon(scalers[i].transform(thistableData),i, D, epsilon, agg_thr, box_thr, models)
        endtime = time.time()
        print(str(D),str(epsilon),str(agg_thr),str(box_thr),"time taken ", (endtime-starttime))
    return models
# def mypredSvm(X, clf):
# #     scores = clf.predict(X)
# #     return scores
#     return np.where(clf.score_samples(X)+clf.intercept_ > -1e-2, 1, -1)

def mypredEpsilon(X, clf):
    scores = clf.predict(np.array(X))
    return scores
    # return np.where(clf.score_samples(X)+clf.intercept_ > -1e-2, 1, -1)

# def testIndiSvm(keyPredg, sstableID, models):
# #     if(keyPredstr == ''):
# #         print('returning cause string is empty')
# #         return 0
# #     if(not isCharacterAscii(keyPredstr)):
# #         print('returning cause not numric')
# #         return 0
# #     keyPred = [str(keyPredstr)]
#     if str(sstableID) in models:
#         # sizeofmodels()
#         clf, reqdim= models[str(sstableID)]
# #         keyPredg = ngramEnforce(keyPred, gramsize, reqdim)
# #         keyPredS = preprocess_test([keyPredg], min_max_scaler)
#         prediction = mypredSvm([keyPredg], clf)
#         return prediction[0]
#     else:
#         print(str(sstableID), ' - sstableid not in models')
#         return 0
# def testIndiSvm2(keyPredg, sstableID, models):
# #     if(keyPredstr == ''):
# #         print('returning cause string is empty')
# #         return 0
# #     if(not isCharacterAscii(keyPredstr)):
# #         print('returning cause not numric')
# #         return 0
# #     keyPred = [str(keyPredstr)]
#     if str(sstableID) in models:
#         # sizeofmodels()
#         clf, reqdim= models[str(sstableID)]
# #         keyPredg = ngramEnforce(keyPred, gramsize, reqdim)
# #         keyPredS = preprocess_test([keyPredg], min_max_scaler)
#         prediction = mypredSvm(keyPredg, clf)
#         return prediction
#     else:
#         print(str(sstableID), ' - sstableid not in models')
#         return 0

def testIndiEpsilon(keyPredg, sstableID, models):
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
        prediction = mypredEpsilon([keyPredg], clf)
        # print(prediction)
        # print(prediction[0])
        return prediction[0]
    else:
        print(str(sstableID), ' - sstableid not in models')
        return 0
def testIndiEpsilon2(keyPredg, sstableID, models):
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
        prediction = mypredEpsilon(keyPredg, clf)
        return prediction
    else:
        print(str(sstableID), ' - sstableid not in models')
        return 0

def putIntoBloomFilters(models):
    bloomfilters = {}
    errorsTrain = {}
    for i in range(0,len(ssTableData)):
        print("bf creating for ",i)
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
            # print(thistableData.tolist())
            falseNegativeAns = testIndiEpsilon2(scalers[i].transform(thistableData), i, models)
            # print(falseNegativeAns.tolist())
            print("--------------------------")
            # print(np.r_['-1',thistableData, np.transpose(np.array([falseNegativeAns]))].tolist())
        #     print(falseNegativeAns)
            numFalseNeg = falseNegativeAns.tolist().count(-1)
        #     newCapacity = int(capacityBloom*1.0*numFalseNeg/len(thistableData))
            newCapacity = max(numFalseNeg/5,1)
            print("newCapacity ",newCapacity)
            f = BloomFilter(capacity=newCapacity, error_rate=errorRate)
            for j in range(0,len(thistableData)):
                if(falseNegativeAns[j]==-1 or removeClassifier):
        #             print("adding ", thistableDataTranspose0[j])
                    f.add(separator.join(map(str,thistableData[j])))
            bloomfilters[i] = f
        errorsTrain[i] = newCapacity
    return (bloomfilters,errorsTrain)

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
                    cAnswer = (testIndiEpsilon(scalers[j].transform(np.array([elem.tolist()]))[0],j, models)==1)
                    # cAnswer = (testIndiEpsilon2(scalers[j].transform([elem]),j, models)==1)
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
                        print(elem)
                        print(elemind)
                        # print(np.array(elem))
                        # print(elem.tolist())
                        # print(len(elem))
                        # print(elem[0])
                        # print(elem[1])
                        # print(type(elem))
                        # print(elem.shape)
                        # print(cAnswer,bfAnswer)
                        # print(testIndiEpsilon(scalers[j].transform(np.array([elem.tolist()]))[0],j, models))
                        # print(testIndiEpsilon2(scalers[j].transform([elem.tolist()]),j, models))
                        # thisanswerscaled = scalers[j].transform([elem.tolist()])
                        # wholetablescaled = scalers[j].transform(thistableData)
                        # falseNegativeAns = testIndiEpsilon2(wholetablescaled, j, models)
                        # print(thisanswerscaled)
                        # print(thisanswerscaled.tolist())
                        # print(testIndiEpsilon2(thisanswerscaled,j,models))
                        # print("----")
                        # print("type of ori data", thistableData.dtype, wholetablescaled.dtype, elem.dtype, thisanswerscaled.dtype)
                        # print(falseNegativeAns.tolist())
                        # print(np.r_['-1',wholetablescaled, np.transpose(np.array([falseNegativeAns]))].tolist())
                        # outfile = open("ssTableErrorEpsilon", "wb")
                        # pkl.dump(thistableData, pickle_out)
                        # pickle_out.close()
                        # np.save("ssTableErrorEpsilonnp", thistableData)
                        raise ValueEroor("not possible")
                    else:
                        #True answer true negative
                        trueNegative+=1
    fprate = falsePositive*1.0/(falsePositive+trueNegative)
    return (fprate,
            truePositive_c,truePositive_bf,truePositive,
            falsePositive_c,falsePositive_bf,falsePositive,
            trueNegative)

def ensemble(D, epsilon, agg_thr, box_thr):
    # print(nu,gamma)
    starttime = time.time()
    models = trainEpsilon(D, epsilon, agg_thr, box_thr)
    persistModel(models,D, epsilon, agg_thr, box_thr)
    # models = loadModel(D, epsilon, agg_thr, box_thr)
    print(D, epsilon, agg_thr, box_thr, "inserting into bloom filters")
    (bloomfilters, errorTrain) = putIntoBloomFilters(models)
    persistBloomFilters(bloomfilters, D, epsilon, agg_thr, box_thr)
    print(D, epsilon, agg_thr, box_thr, "performing tests")
    (fprate,
    truePositive_c,truePositive_bf,truePositive,
    falsePositive_c,falsePositive_bf,falsePositive,
    trueNegative) = performTests(models, bloomfilters)
    observations = (errorTrain, fprate,
                    truePositive_c,truePositive_bf,truePositive,
                    falsePositive_c,falsePositive_bf,falsePositive,
                    trueNegative)
    print(fprate)
    persistObservations(observations, D, epsilon, agg_thr, box_thr)
    endtime = time.time()
    return (endtime-starttime, D, epsilon, agg_thr, box_thr)



gramsize = -1
maxiter = 100
# models = {} # storing models based on sstable ids
# nu = 0.1
# gamma = 1
removeClassifier = False


# In[19]:


# thres = -0.1
# bloomfilters = {}
capacityBloom = 30000
errorRate = 0.1
skipFactor = 50


# In[20]:


ssTableData = {}
separator = ":"
namePrefix = "capacityby5_epsilon"
# namePrefixOri = "6nu6gamma_ex4"
namePrefixOri = namePrefix

(ssTableData, alltemptableData) = getData()
ssTableData, dimlist = createNgrams(ssTableData, alltemptableData,4,4)
scalers = getScalers()
# nus = np.logspace(-3,0,6)
# gammas = np.logspace(-3,3,6)
# nus = np.logspace(-3,0,6)
# gammas = np.logspace(-1,3,6)
num_cores = multiprocessing.cpu_count()

# tempresults = ensemble(6,1e-2,0.9,1e-2)
# print(tempresults)
Ds = map(int,np.logspace(math.log10(2),math.log10(256),5))
epsilons = np.logspace(-4,-1,5)
agg_thrs = np.linspace(0.8,1.0,3)
box_thrs = np.logspace(-4,-1,5)
results = Parallel(n_jobs=num_cores)(delayed(ensemble)(i,j,k,l) for i,j,k,l in tqdm(product(Ds, epsilons, agg_thrs, box_thrs)))
print(results)
