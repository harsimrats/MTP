import numpy as np
import matplotlib.pyplot as plt
# %matplotlib inline
from pybloom import BloomFilter
import inspect
import matplotlib.font_manager
from sklearn import svm
import os
import glob
import pickle as pkl
import time
# %matplotlib notebook
from itertools import product
from joblib import Parallel, delayed
import multiprocessing
from tqdm import tqdm
import sys
import json
ssTableData = {}
separator = ":"
namePrefix = sys.argv[3]
namePrefixOri = namePrefix


takesample = False

typedata = sys.argv[1]
# typedata = "vdisk"
if typedata=="vdisk":
    import vdiskHelper as dataHelper
typemodel = sys.argv[2]
if typemodel=="epsilon":
    import epsilonHelper as modelHelper
elif typemodel=="svmsimple":
    import svmsklearncoreHelper as modelHelper
elif typemodel=="libsvmsch":
    import svmlibsvmHelper as modelHelper
elif typemodel=="isolationforest":
    import isolationForestHelper as modelHelper
elif typemodel=="np":
    import npHelper as modelHelper
elif typemodel=="dic":
    import dicHelper as modelHelper
elif typemodel == "alwaysNegative":
    import alwaysNegative as modelHelper

if __name__=="__main__":
    results = dataHelper.get(lessData=True)
    oriSsTableDataStr, allSsTableDataStr, bagOfWordsModel = results
    # (oriSsTableData, alltemptableData, ssTableData, oriSsTableDataStrSet, vectorizer, sstableBagData) = dataHelper.get()
    totaldatapoints = 0
    for i in oriSsTableDataStr:
        totaldatapoints += len(oriSsTableDataStr[i])
    print("totaldatapoints are ", totaldatapoints)
    # scalers = getScalers()

    def loadObservations(params):
        pickle_out = open("Observations/observations_"+namePrefix+modelHelper.paramsToString(params),"rb")
        obs = pkl.load(pickle_out)
        pickle_out.close()
        return obs
    def loadBloomFilters(params):
        pickle_out = open("bloomfilters/bloomfilters_"+namePrefix+modelHelper.paramsToString(params),"rb")
        bloomfilters = pkl.load(pickle_out)
        pickle_out.close()
        return bloomfilters
    def sumsizeofbloom(bloomfilters):
        agg =0
        for f in bloomfilters.values():
            agg += f.num_bits
        agg = agg/8
        return agg

    if takesample:

        obs = loadObservations(modelHelper.sampleparams)
        print(obs)
        bloomfilters = loadBloomFilters(modelHelper.sampleparams)
        tempsize = sumsizeofbloom(bloomfilters)
        tempsum = 0
        for k in obs["errorTrain"]:
            tempsum += obs["errorTrain"][k]
        obsdic2 = {"fpr": obs["fpr"], "falsenegative_c": tempsum, "falsepositive_c": obs["falsepositive_c"],
                              "falsepositive_bf": obs["falsepositive_bf"], "totalDataPoints": totaldatapoints,
                              "bfsize": tempsize
                            }
        for metaparam in modelHelper.metaparams:
            obsdic2[metaparam] = modelHelper.loadMetaParam(namePrefix, modelHelper.sampleparams,
                                                                      metaparam)

        f = open("Observations/compiledobs_" + namePrefix + ".pkl", "wb")
        pkl.dump(obsdic2, f)
        f.close()
        jsontext = json.dumps(obsdic2)
        print(jsontext)
        f = open("Observations/json_" + namePrefix + ".json", "w")
        f.write(jsontext)
        f.close()
    else:
        mshape = modelHelper.getMshape()
        print(mshape)

        obsdic = {}
        obsdic2 = {}
        fprates = np.zeros(mshape)
        fprateslist = []
        falsenegatives_c = []
        falsenegatives_c_validation = []
        falsepositive_c = []
        falsepositive_bf = []
        bfsizelist = []
        modelSizeList = []
        fpdic = {}
        for tuplethis in modelHelper.getProductParams():
            print(tuplethis)
            try:
                obs = loadObservations(modelHelper.createParams(tuplethis))
                fprateslist.append(obs["fpr"])
                tempsum = 0
                for k in obs["errorTrain"]:
                    tempsum += obs["errorTrain"][k]
                falsenegatives_c.append(tempsum)
                falsepositive_c.append(obs["falsepositive_c"])
                falsepositive_bf.append(obs["falsepositive_bf"])
        #         tempsum = 0
        #         for k in obs[9]:
        #             tempsum += obs[9][k]
        #         falsenegatives_c_validation.append(tempsum)
                fpdic[tuplethis] = obs["fpr"]
                obsdic2[tuplethis] = {"fpr":obs["fpr"], "falsenegative_c": tempsum, "falsepositive_c":obs["falsepositive_c"], "falsepositive_bf": obs["falsepositive_bf"], "totalDataPoints": totaldatapoints}
                print(obs["fpr"])
            except IOError:
                obs = None
                print("can't find ", modelHelper.createParams(tuplethis))
                fprateslist.append(1)
                fpdic[tuplethis] = None
                print(obs)
            try:
                bloomfilters = loadBloomFilters(modelHelper.createParams(tuplethis))
                tempsize = sumsizeofbloom(bloomfilters)
                bfsizelist.append(tempsize)
                obsdic2[tuplethis]["bfsize"] = tempsize
                print(tempsize)
            except IOError:
                print("can't find ", modelHelper.createParams(tuplethis))
                tempsize = None
                bfsizelist.append(1e6)
                print(tempsize)
            try:
                for metaparam in modelHelper.metaparams:
                    obsdic2[tuplethis][metaparam] = modelHelper.loadMetaParam(namePrefix,modelHelper.createParams(tuplethis), metaparam)
            except IOError:
                print("can't find ", modelHelper.createParams(tuplethis))
                for metaparam in modelHelper.metaparams:
                    obsdic2[tuplethis][metaparam] = None
            obsdic[tuplethis] = obs

        #     try:
        #         models = loadModel(i,j)
        #         tempsize = sumsizeofmodelssvmReal(models)
        #         modelSizeList.append(tempsize)
        #         print(tempsize)
        #     except IOError:
        #         tempsize = None
        #         modelSizeList.append(1e10)
        #         print(tempsize)

        #     fprateslist.append(obs[1])
        #     fpdic[(i,j)] = obs[1]


        fprateslist = np.array(fprateslist).reshape(mshape)
        falsenegatives_c = np.array(falsenegatives_c).reshape(mshape)
        falsepositive_c = np.array(falsepositive_c).reshape(mshape)
        falsepositive_bf = np.array(falsepositive_bf).reshape(mshape)
        bfsizelist = np.array(bfsizelist).reshape(mshape)
        # falsenegatives_c_validation = np.array(falsenegatives_c_validation).reshape(mshape)
        # modelSizeList = np.array(modelSizeList).reshape(mshape)
        # totalSizeList = bfsizelist + modelSizeList

        wholejson = list()
        for i in obsdic2:
            thisparams = modelHelper.createParams(i)
            for j, k in obsdic2[i].items():
                thisparams[j] = k
            wholejson.append(thisparams)
        grammerjson = dict()
        grammerjson["data"] = wholejson
        # print(str(wholejson))
        f = open("Observations/compiledobs_" + namePrefix + ".pkl", "wb")
        pkl.dump(grammerjson, f)
        f.close()
        jsontext = json.dumps(grammerjson)
        print(jsontext)
        f = open("Observations/json_" + namePrefix + ".json", "w")
        f.write(jsontext)
        f.close()

    # excelstr =  modelHelper.paramnames + ", falsepositive_c, falsepositive_bf, falsenegative_c, bfsize\n"
    # for i in obsdic2:
    #   tempstr = [str(i[j]) for j in range(len(modelHelper.sampleparams))]
    #   tempstr = ",".join(tempstr)
    #     excelstr += tempstr +","+str(obsdic2[i]["fpr"])+","+str(obsdic2[i]["falsepositive_c"])+","+str(obsdic2[i]["falsepositive_bf"])+","+str(obsdic2[i]["falsenegative_c"])+","+str(obsdic2[i]["bfsize"])+"\n"
    # f = open(namePrefix+"excel.csv","w")
    # print(excelstr)
    # f.write(excelstr)
    # f.close()

    # print(obsdic2)











