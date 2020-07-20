import vdiskHelper as dataHelper
import epsilonHelper as modelHelper
import time
from BagOfWordsv1 import BagOfWordsv1
class Model:
    def __init__(self, model, bagOfWodsModel):
        self.model = model
        self.bagOfWordsModel = bagOfWodsModel
    def mypred(self, X):
        predictions = modelHelper.predict(self.model,X)
        return predictions
class Interface:
    def __init__(self, lessData=False):
        self.lessData =lessData
        return

    def getmodel(self, keys, columnfamily):
        # returning trained model on the keys given as argument
        # keys are not altered, so I am specifying column family so that you can parse key accordingly
        # keys will be array or list of strings
        # columnfamily will be string
        bagOfWordsModel =BagOfWordsv1(keys, 4,4)
        params = modelHelper.sampleparams
        baggedData = bagOfWordsModel.process(keys)
        clf = modelHelper.initAndFit(params, baggedData)
        return Model(clf, bagOfWordsModel)



    def predict(self, keys, model):
        # given a key and a model return the prediction of model on that key
        # key will be string
        # model will be of same structure as returned by getmodel()
        thisbaggeddata = model.bagOfWordsModel.process(keys)
        prediction = model.mypred(thisbaggeddata)
        return prediction
    def getdata(self, columnfamily):
        results = dataHelper.get(lessData=self.lessData)
        return results

if __name__=="__main__":
    interface = Interface(lessData=True)
    # Get data in simple string format like below
    wholedata = interface.getdata("vdisk")[0]
    # lets take keys like below
    keys = wholedata[0]
    subsetkeys = keys[4:40]

    # Get all sstables models like below
    model = interface.getmodel(keys, "vdisk")

    #suppose we want to use this for model made for sstable 0
    predictions = interface.predict(subsetkeys, model)
    print(predictions)