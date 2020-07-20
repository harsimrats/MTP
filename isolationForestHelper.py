sampleparams = {}
from sklearn.ensemble import IsolationForest
import numpy as np
import math
from itertools import product
import pickle as pkl
def initAndFit(params, nplst1):
	clf = IsolationForest(random_state=0).fit(nplst1)
	return clf
def predict(clf, X):
	return np.array(clf.predict(X))
def paramsToString(params):
	return str("")
def createParams(tuplethis):
	params = {}
	return params
def getProductParams():
	return product()
def mshape():
	return ()
def persistModel(namePrefix,models, params):
    pickle_out = open("svmModels/svmmodelstandardscaled_"+namePrefix+paramsToString(params),"wb")
    pkl.dump(models, pickle_out)
    pickle_out.close()


# In[13]:


def loadModel(namePrefixOri, params):
    pickle_out = open("svmModels/svmmodelstandardscaled_"+namePrefixOri+paramsToString(params),"rb")
    models = pkl.load(pickle_out)
    pickle_out.close()
    return models