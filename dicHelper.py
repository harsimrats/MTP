sampleparams = {}
import numpy as np
import math
from itertools import product
import pickle as pkl
import scipy
separator = ":"
def initAndFit(params, nplst1):
    dic ={}
    # print(np.array(nplst1))
    if isinstance(nplst1, scipy.sparse.csr.csr_matrix):
        nplst1 = nplst1.todense()
    tempnplst1 = np.array(nplst1,dtype=np.float32)
    for i in tempnplst1:
        tempstr = separator.join(map(str,i))
        dic[tempstr] = True
    # print(dic.keys())
    return dic
def predict(clf, X):
    if isinstance(X, scipy.sparse.csr.csr_matrix):
        X = X.todense()
    Xtest = np.array(X, dtype=np.float32)
    return np.array([1 if (separator.join(map(str,Xtest[i])) in clf) else -1 for i in range(len(Xtest))])
    # return np.array([-1 for i in range(len(X))])
def paramsToString(params):
    return ""
def createParams(tuplethis):
    params = {}
    return params
def getProductParams():
    return product()

def getMshape():
    return ()

def persistModel(namePrefix, models, params):
    pickle_out = open("svmModels/svmmodelstandardscaled_"+namePrefix+paramsToString(params),"wb")
    print("gonna persist model ----------")
    pkl.dump(models, pickle_out)
    pickle_out.close()


# In[13]:


def loadModel(namePrefixOri,params):
    pickle_out = open("svmModels/svmmodelstandardscaled_"+namePrefixOri+paramsToString(params),"rb")
    models = pkl.load(pickle_out)
    pickle_out.close()
    return models
def loadMetaParam(namePrefix, params, metaparam):
    models = loadModel(namePrefix, params)
    if metaparam=="size":
        temp = 0
        for m in models:
            tempmodel = models[m][0]
            temp += len(tempmodel)
        return temp