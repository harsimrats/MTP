sampleparams = {"D":9, "epsilon":1e-2, "agg_thr":0.9, "box_thr":1e-2}
paramnames = "Dimension, epsilon, agg_thr, box_thr, fpr"
metaparams = {
                "size"
                }
import epsilon_occ as ep
import numpy as np
import math
from itertools import product
import pickle as pkl
def initAndFit(params, nplst1):
    try:
        clf = ep.RPEOCC(params["D"], params["epsilon"], params["agg_thr"], params["box_thr"])
        # nplst1 = nplst1.todense()
        # print("------training")
        # print(nplst1)
        # print(type(nplst1))
        # print("-----dense")
        # print(nplst1.todense())
        # print(type(nplst1.todense()))
        clf.fit(nplst1)
        # clf.fit(np.array(nplst1.todense()))
        return clf
    except Exception as e:
        f = open("wrongdata.pkl", "wb")
        pkl.dump(nplst1.todense(), f)
        f.close()  
        print(e)
        raise
def predict(clf, X):
    # X = X.todense()
    # return clf.predict(np.array(X))
    return clf.predict(X)
def paramsToString(params):
    return str(params["D"])+"_"+str(params["epsilon"])+"_"+str(params["agg_thr"])+"_"+str(params["box_thr"])
def createParams(tuplethis):
    params = {"D":tuplethis[0], "epsilon":tuplethis[1], "agg_thr":tuplethis[2], "box_thr":tuplethis[3]}
    return params
def getProductParams():
    Ds = map(int,np.logspace(math.log10(2),math.log10(256),10))
    epsilons = np.logspace(-4,-1,10)
    agg_thrs = np.linspace(0.8,1.0,3)
    box_thrs = np.append(0,np.logspace(-4,-1,9))
    # box_thrs = np.logspace(-4,-1,3)
    return product(Ds, epsilons, agg_thrs, box_thrs)

def getMshape():
    Ds = list(map(int,np.logspace(math.log10(2),math.log10(256),10)))
    epsilons = np.logspace(-4,-1,10)
    agg_thrs = np.linspace(0.8,1.0,3)
    box_thrs = np.append(0,np.logspace(-4,-1,9))
    return (len(Ds), len(epsilons), len(agg_thrs), len(box_thrs))

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
            temp += tempmodel.get_size()
        return temp