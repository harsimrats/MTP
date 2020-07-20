sampleparams = {}
import numpy as np
import math
from itertools import product
import pickle as pkl
def initAndFit(params, nplst1):
	clf = nplst1.tolist()
	return clf
def predict(clf, X):
	Xlist = X.tolist()
	return [1 if (Xlist[i] in clf) else -1 for i in range(len(X))]
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