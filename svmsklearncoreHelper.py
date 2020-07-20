sampleparams = {"nu":0.1, "gamma":100, "kernel":"rbf","tol": 0.001}
metaparams = {
    "size",
    "nsv"
}
from sklearn import svm
import numpy as np
from itertools import product
import pickle as pkl
import scipy
def initAndFit(params, nplst1):
    clf = svm.OneClassSVM(nu=params["nu"], kernel=params["kernel"], gamma=params["gamma"], tol=params["tol"])
    clf.fit(nplst1)
    return clf
def predict(clf, X):
    tmp = clf.predict(X)
    # print("tmp is ----")
    # print(tmp)
    return tmp
def paramsToString(params):
    return str(params["nu"])+"_"+str(params["gamma"])+"_"+str(params["kernel"]) + "_" +str(params["tol"])
def createParams(tuplethis):
    params = {"nu":tuplethis[0], "gamma":tuplethis[1], "kernel":"rbf", "tol": tuplethis[2]}
    return params
def getProductParams():
    nus = np.logspace(-3,0,4)
    gammas = np.logspace(-1,3,3)
    tols = [0.001, 0.01, 0.1, 1.0]
    return product(nus, gammas,tols)
def getMshape():
    nus = np.logspace(-3,0,4)
    gammas = np.logspace(-1,3,3)
    tols = [0.001, 0.01, 0.1, 1.0]
    return (len(nus), len(gammas), len(tols))
def persistModel(namePrefix,models, params):
    pickle_out = open("svmModels/svmmodelstandardscaled_"+namePrefix+paramsToString(params),"wb")
    pkl.dump(models, pickle_out)
    pickle_out.close()
# for i in models:
# 	svm_save_model("svmModels/svmmodelstandardscaled_"+namePrefix+paramsToString(params)+"_"+ str(i), models[i][0])


# In[13]:


def loadModel(namePrefix, params):
    pickle_out = open("svmModels/svmmodelstandardscaled_"+namePrefix+paramsToString(params),"rb")
    models = pkl.load(pickle_out)
    pickle_out.close()
    return models

def sumsizeofmodelssvmReal(models):
    agg =0
    for sstableID in models:
        tempmodel, reqdim= models[str(sstableID)]
        # tempsize = 8*(tempmodel.support_vectors_.shape[0]*tempmodel.support_vectors_.shape[1])+ 8*(tempmodel.dual_coef_.shape[0]*tempmodel.dual_coef_.shape[1]) + 4*(tempmodel.support_.shape[0])
        tempsize = 0
        if isinstance(tempmodel.support_vectors_, scipy.sparse.csr.csr_matrix):
            tempsize += tempmodel.support_vectors_.data.nbytes
        else:
            tempsize += tempmodel.support_vectors_.nbytes
        if isinstance(tempmodel.dual_coef_, scipy.sparse.csr.csr_matrix):
            tempsize += tempmodel.dual_coef_.data.nbytes
        else:
            tempsize += tempmodel.dual_coef_.nbytes
        tempsize += tempmodel.support_.nbytes
        agg += tempsize
    # print(str(tempsize))
    # print(sys.getsizeof(p))
    # print(sstableID, tempsize)
    return agg


def loadMetaParam(namePrefix, params, metaparam):
    models = loadModel(namePrefix, params)
    if metaparam=="size":
        return sumsizeofmodelssvmReal(models)
    elif metaparam=="nsv":
        temp = 0
        for m in models:
            tempmodel = models[m][0]
            if isinstance(tempmodel.support_vectors_, scipy.sparse.csr.csr_matrix):
                temp += tempmodel.support_vectors_.shape[0]
            else:
                temp += len(tempmodel.support_vectors_)
        return temp