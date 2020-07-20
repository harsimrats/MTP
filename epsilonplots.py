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

def sumsizeofbloom(bloomfilters):
    agg =0
    for f in bloomfilters.values():
        agg += f.num_bits
    agg = agg/8
    return agg


def sumsizeofmodelsepsilonReal(models):
    agg =0
    for sstableID in models:
        tempmodel, reqdim= models[str(sstableID)]
        # tempsize = 8*(tempmodel.support_vectors_.shape[0]*tempmodel.support_vectors_.shape[1])+ 8*(tempmodel.dual_coef_.shape[0]*tempmodel.dual_coef_.shape[1]) + 4*(tempmodel.support_.shape[0])
        tempsize = tempmodel.get_size()/8.0
        agg += tempsize
        # print(str(tempsize))
        # print(sys.getsizeof(p))
        print(sstableID, tempsize)
    return agg
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


def loadBloomFilters(D, epsilon, agg_thr, box_thr):
    pickle_out = open("bloomfilters/bloomfilters_"+namePrefix+str(D)+"_"+str(epsilon)+"_"+str(agg_thr)+"_"+str(box_thr),"rb")
    bloomfilters = pkl.load(pickle_out)
    pickle_out.close()
    return bloomfilters

# In[15]:


def persistObservations(obs, D, epsilon, agg_thr, box_thr):
    print(obs, D, epsilon, agg_thr, box_thr)
    pickle_out = open("Observations/observations_"+namePrefix+str(D)+"_"+str(epsilon)+"_"+str(agg_thr)+"_"+str(box_thr),"wb")
    pkl.dump(obs, pickle_out)
    pickle_out.close()


def loadObservations(D, epsilon, agg_thr, box_thr):
    pickle_out = open("Observations/observations_"+namePrefix+str(D)+"_"+str(epsilon)+"_"+str(agg_thr)+"_"+str(box_thr),"rb")
    obs = pkl.load(pickle_out)
    pickle_out.close()
    return obs

# In[16]:


def persistTimes(obs):
    pickle_out = open("times/times_"+namePrefix,"wb")
    pkl.dump(obs, pickle_out)
    pickle_out.close()

def loadTimes():
    pickle_out = open("times/times_"+namePrefix,"rb")
    obs = pkl.load(pickle_out)
    pickle_out.close()
    return obs

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
namePrefix = "6nu6gamma_epsilon"
# namePrefixOri = "6nu6gamma_ex4"
namePrefixOri = namePrefix

(ssTableData, alltemptableData) = getData()
ssTableData, dimlist = createNgrams(ssTableData, alltemptableData,4,4)
scalers = getScalers()
# nus = np.logspace(-3,0,6)
# gammas = np.logspace(-3,3,6)
# nus = np.logspace(-3,0,6)
# gammas = np.logspace(-1,3,6)
# num_cores = multiprocessing.cpu_count()

# tempresults = ensemble(6,1e-2,0.9,1e-2)
# print(tempresults)
Ds = map(int,np.logspace(math.log10(2),math.log10(256),5))
epsilons = np.logspace(-4,-1,5)
agg_thrs = np.linspace(0.8,1.0,3)
box_thrs = np.logspace(-4,-1,5)
# results = Parallel(n_jobs=num_cores)(delayed(ensemble)(i,j,k,l) for i,j,k,l in tqdm(product(Ds, epsilons, agg_thrs, box_thrs)))
# print(results)
mshape = (5,5,3,5)

obsdic = {}
fprates = np.zeros(mshape)
fprateslist = []
bfsizelist = []
modelSizeList = []
fpdic = {}
for i,j,k,l in product(Ds, epsilons, agg_thrs, box_thrs):
    print(i,j,k,l)
    try:
        obs = loadObservations(i,j,k,l)
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
        tempsize = sumsizeofmodelsepsilonReal(models)
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

minval = 1.0
for i in obsdic:
    print(i, obsdic[i][1])
    if obsdic[i][1] < minval:
        minval = obsdic[i][1]
        minpair =i

