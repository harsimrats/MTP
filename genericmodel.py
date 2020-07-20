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
import scipy







# In[12]:


# def persistModel(models, D, epsilon, agg_thr, box_thr):
#     pickle_out = open("svmModels/svmmodelstandardscaled_"+namePrefix+str(D)+"_"+str(epsilon)+"_"+str(agg_thr)+"_"+str(box_thr),"wb")
#     pkl.dump(models, pickle_out)
#     pickle_out.close()




# In[14]:


def persistBloomFilters(bloomfilters, params):
    pickle_out = open("bloomfilters/bloomfilters_"+namePrefix+modelHelper.paramsToString(params),"wb")
    pkl.dump(bloomfilters, pickle_out)
    pickle_out.close()

def loadBloomFilters(params):
    pickle_out = open("bloomfilters/bloomfilters_"+namePrefix+modelHelper.paramsToString(params),"rb")
    bloomfilters = pkl.load(pickle_out)
    pickle_out.close()
    return bloomfilters


# In[15]:


def persistObservations(obs, params):
    pickle_out = open("Observations/observations_"+namePrefix+modelHelper.paramsToString(params),"wb")
    pkl.dump(obs, pickle_out)
    pickle_out.close()


# In[16]:


def persistTimes(obs):
    pickle_out = open("times/times_"+namePrefix,"wb")
    pkl.dump(obs, pickle_out)
    pickle_out.close()

def getScalers():
    scalers = {}
    for i in range(0,len(ssTableData)):
        thistableData = ssTableData[i]
        scaler = preprocessing.StandardScaler().fit(thistableData)
        scalers[i] = scaler
    return scalers






def mypred(X, clf):
    predictions = modelHelper.predict(clf, X)
    # predictions = clf.predict(np.array(X))
    return predictions

def testIndi(keyPredg, sstableID, models):
    if str(sstableID) in models:
        clf, reqdim= models[str(sstableID)]
        # print("----------strange")
        # print(type(keyPredg))
        # print(keyPredg)
        # print(keyPredg.toarray())
        # print(type([keyPredg]))
        if isinstance(keyPredg, scipy.sparse.csr.csr_matrix):
            # Bracket is removed for sparse case
            prediction = mypred(keyPredg, clf)
        else:
            prediction = mypred(np.array([keyPredg]), clf)
        return prediction[0]
    else:
        print(str(sstableID), ' - sstableid not in models')
        return 0
def testIndi2(keyPredg, sstableID, models):
    if str(sstableID) in models:
        clf, reqdim= models[str(sstableID)]
        prediction = mypred(keyPredg, clf)
        return prediction
    else:
        print(str(sstableID), ' - sstableid not in models')
        return 0

def putIntoBloomFilters(models, oriSsTableDataStr, bagOfWordsModel):
    bloomfilters = {}
    errorsTrain = {}
    for i in range(0,len(oriSsTableDataStr)):
        print("bf creating for ",i)
        orithistableData = oriSsTableDataStr[i]
        thisbaggeddata = bagOfWordsModel.process(orithistableData)
        # thistableDataTranspose0 = thistableData.transpose()[0]
        if removeClassifier:
            newCapacity = max(len(orithistableData),1)
            print("newCapacity ",newCapacity)
            f = BloomFilter(capacity=newCapacity, error_rate=errorRate)
            for j in range(0,len(orithistableData)):
                f.add(orithistableData[i])
            bloomfilters[i] = f
        else:
            # print(thistableData.tolist())
            # print(thisbaggeddata)
            # print(type(thisbaggeddata))
            # print("--")
            # print(thisbaggeddata.todense())
            falseNegativeAns = testIndi2(thisbaggeddata, i, models)
            # print(falseNegativeAns.tolist())
            print("--------------------------")
            # print(np.r_['-1',thistableData, np.transpose(np.array([falseNegativeAns]))].tolist())
        #     print(falseNegativeAns)
            numFalseNeg = falseNegativeAns.tolist().count(-1)
        #     newCapacity = int(capacityBloom*1.0*numFalseNeg/len(thistableData))
            newCapacity = numFalseNeg
            bloomCapacity = max(newCapacity/1,1)
            print("newCapacity ",newCapacity)
            f = BloomFilter(capacity=bloomCapacity, error_rate=errorRate)
            for j in range(0,len(orithistableData)):
                if(falseNegativeAns[j]==-1 or removeClassifier):
        #             print("adding ", thistableDataTranspose0[j])
                    f.add(orithistableData[j])
            bloomfilters[i] = f
        errorsTrain[i] = newCapacity
    return (bloomfilters,errorsTrain)

def performTests(models, bloomfilters, oriSsTableDataStr, bagOfWordsModel, skipFactor):
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
    setsOfStringData = [set(oriSsTableDataStr[i]) for i in range(len(oriSsTableDataStr))]
    for i in range(0, len(oriSsTableDataStr)):
        print("on sstable number ", i)
        orithistableData = oriSsTableDataStr[i]
        thisbaggeddata = bagOfWordsModel.process(orithistableData)
        # thistableDataTranspose0 = thistableData.transpose()[0]
        for elemind in range(0,len(orithistableData),skipFactor):
            elemOri = orithistableData[elemind]
            elemBagged = thisbaggeddata[elemind]
            for j in range(0,len(oriSsTableDataStr)):
                oriThisSsTableDataStrSet = setsOfStringData[j]
                if removeClassifier:
                    cAnswer = False
                else:
                    cAnswer = (testIndi(elemBagged,j, models)==1)
                    # cAnswer = (testIndiEpsilon2(scalers[j].transform([elem]),j, models)==1)
    #             print(cAnswer)
                if not cAnswer:
                    bfAnswer = elemOri in bloomfilters[j]
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
                        # elemOriStr = ":".join(map(str,map(int, elemOri)))
                        if elemOri in oriThisSsTableDataStrSet:
                            truePositive+=1
                            if bfAnswer is None:
                                truePositive_c+=1
                            else:
                                truePositive_bf+=1
                            break
                        else:
                            falsePositive+=1
                            if bfAnswer is None:
                                falsePositive_c+=1
                            else:
                                falsePositive_bf+=1
                else:
                    if(i==j):
                        raise ValueError("not possible")
                    else:
                        #True answer true negative
                        trueNegative+=1
    fprate = falsePositive*1.0/(falsePositive+trueNegative)
    print("fpr is ", fprate)
    return (fprate,
            truePositive_c,truePositive_bf,truePositive,
            falsePositive_c,falsePositive_bf,falsePositive,
            trueNegative)

def trainIndi(nplst1, sstableID, params, models):
    # print("passed mean was ",np.mean(nplst1, axis=0))
    # print("passed sigma was ",np.var(nplst1, axis=0))
    clf = modelHelper.initAndFit(params, nplst1)
    reqdim = nplst1[0].shape[0]
#     print("reqdim is ",reqdim)
    models[str(sstableID)] = (clf, reqdim)
    print('training done with sstableid - ', sstableID)
    return models

def trainFn(params, oriSsTableDataStr, bagOfWordsModel):
    # train the model on generated data
    models = {}
    for i in range(0,len(oriSsTableDataStr)):
        starttime = time.time()
        # thistableData = ssTableData[i]
        baggedData = bagOfWordsModel.process(oriSsTableDataStr[i])
        models = trainIndi(baggedData,i, params, models)
        endtime = time.time()
        print(modelHelper.paramsToString(params),"time taken ", (endtime-starttime))
    return models

createnew = True

def ensemble(params, oriSsTableDataStr, bagOfWordsModel, skipFactor):
    # print(nu,gamma)
    starttime = time.time()
    if createnew:
        models = trainFn(params, oriSsTableDataStr, bagOfWordsModel)
        modelHelper.persistModel(namePrefix,models,params)
        print(params, "inserting into bloom filters")
        (bloomfilters, errorTrain) = putIntoBloomFilters(models, oriSsTableDataStr, bagOfWordsModel)
        persistBloomFilters(bloomfilters, params)
    else:
        models = modelHelper.loadModel(namePrefix, params)
        bloomfilters = loadBloomFilters(params)
    print(params, "performing tests")
    (fprate,
    truePositive_c,truePositive_bf,truePositive,
    falsePositive_c,falsePositive_bf,falsePositive,
    trueNegative) = performTests(models, bloomfilters, oriSsTableDataStr, bagOfWordsModel, skipFactor)
    observations = (errorTrain, fprate,
                    truePositive_c,truePositive_bf,truePositive,
                    falsePositive_c,falsePositive_bf,falsePositive,
                    trueNegative)
    observationsdic = params.copy()
    observationsdic["errorTrain"] = errorTrain
    observationsdic["fpr"] = fprate
    observationsdic["truepositive_c"] = truePositive_c
    observationsdic["truepositive_bf"] = truePositive_bf
    observationsdic["truepositive"] = truePositive
    observationsdic["falsepositive_c"] = falsePositive_c
    observationsdic["falsepositive_bf"] = falsePositive_bf
    observationsdic["falsepositive"] = falsePositive
    observationsdic["truenegative"] = trueNegative
    persistObservations(observationsdic, params)
    endtime = time.time()
    return (endtime-starttime, params)



# models = {} # storing models based on sstable ids
# nu = 0.1
# gamma = 1
removeClassifier = False


# In[19]:


# thres = -0.1
# bloomfilters = {}
errorRate = 0.3


# In[20]:



# In[20]:

if __name__ == '__main__':
    typedata = "vdisk"
    # typedata = sys.argv[1]
    if typedata == "vdisk":
        import vdiskHelper as dataHelper
    # typemodel = "svmsimple"
    # typemodel = "dic"
    typemodel = sys.argv[2]
    if typemodel == "epsilon":
        import epsilonHelper as modelHelper
    elif typemodel == "svmsimple":
        import svmsklearncoreHelper as modelHelper
    elif typemodel == "libsvmsch":
        import svmlibsvmHelper as modelHelper
    elif typemodel == "isolationforest":
        import isolationForestHelper as modelHelper
    elif typemodel == "np":
        import npHelper as modelHelper
    elif typemodel == "dic":
        import dicHelper as modelHelper
    elif typemodel == "alwaysNegative":
        import alwaysNegative as modelHelper
    # ssTableData = {}
    # namePrefix = "svmsimple"
    # namePrefix = "dic"
    namePrefix = sys.argv[3]
    # namePrefixOri = "6nu6gamma_ex4"
    namePrefixOri = namePrefix
    lessData = True
    if lessData:
        skipFactor = 1
    else:
        skipFactor = 50
    results = dataHelper.get(lessData=lessData)
    oriSsTableDataStr, allSsTableDataStr, bagOfWordsModel = results
    # print(sstableBagData)

    # scalers = getScalers()

    # nus = np.logspace(-3,0,6)
    # gammas = np.logspace(-3,3,6)
    # nus = np.logspace(-3,0,6)
    # gammas = np.logspace(-1,3,6)
    num_cores = multiprocessing.cpu_count()

    tempresults = ensemble(modelHelper.sampleparams, oriSsTableDataStr, bagOfWordsModel, skipFactor)
    print(tempresults)
    # results = Parallel(n_jobs=num_cores)(delayed(ensemble)(modelHelper.createParams(tuplethis), oriSsTableDataStr, bagOfWordsModel, skipFactor) for tuplethis in tqdm(modelHelper.getProductParams()))
    # print(results)
    # persistTimes(results)