sampleparams = {}
metaparams = {}
import numpy as np
from itertools import product
import pickle as pkl
separator = ":"
def initAndFit(params, nplst1):
    return None
def predict(clf, X):
    return np.array([-1 for x in range(X.shape[0])])
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
    if metaparam=="size":
        temp = 0
        return temp