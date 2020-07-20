sampleparams = {"nu":0.1, "gamma":0.1, "kernel":"rbf", "tol": 0.001}
import sys
from os.path import expanduser
home = expanduser("~")
sys.path.insert(1, home + '/Downloads/libsvm-3.24/python')
from svmutil import *
import numpy as np
import math
from itertools import product
import pickle as pkl
def initAndFit(params, nplst1):
	# nplst1 = nplst1.todense()
	y = [1 for i in range(nplst1.shape[0])]
	prob = svm_problem(y, nplst1)
	param = '-s 2 '+'-n '+str(params["nu"])+" -g "+str(params["gamma"]) + " -e "+str(params["tol"])
	print("param is ", param)
	svmparams = svm_parameter(param)
	print(svmparams)
	clf = svm_train(prob, svmparams)
	return clf
def predict(clf, X):
	return np.array(svm_predict([], X, clf)[0])
def paramsToString(params):
	return str(params["nu"])+"_"+str(params["gamma"])+"_"+str(params["kernel"])+ "_"+ str(params["tol"])
def createParams(tuplethis):
	params = {"nu":tuplethis[0], "gamma":tuplethis[1], "kernel":"rbf", "tol": tuplethis[2]}
	return params
def getProductParams():
	nus = np.logspace(-3,0,4)
	gammas = np.logspace(-1,3,3)
	tols = [0.001, 0.01, 0.1, 1.0]
	return product(nus, gammas, tols)
def getMshape():
	nus = np.logspace(-3,0,4)
	gammas = np.logspace(-1,3,3)
	tols = [0.001, 0.01, 0.1, 1.0]
	return (len(nus), len(gammas), len(tols))
def persistModel(namePrefix,models, params):
    # pickle_out = open("svmModels/svmmodelstandardscaled_"+namePrefix+modelHelper.paramsToString(params),"wb")
    # pkl.dump(models, pickle_out)
    # pickle_out.close()
	for i in models:
		svm_save_model("svmModels/svmmodelstandardscaled_"+namePrefix+paramsToString(params)+"_"+ str(i), models[i][0])


# In[13]:


def loadModel(namePrefix, params):
    # pickle_out = open("svmModels/svmmodelstandardscaled_"+namePrefixOri+modelHelper.paramsToString(params),"rb")
    # models = pkl.load(pickle_out)
    # pickle_out.close()
    # return models
	models = {}
	for i in range(24):
		m = svm_load_model("svmModels/svmmodelstandardscaled_"+namePrefix+paramsToString(params)+"_"+ str(i))
		models[str(i)] = (m,0)
	return models