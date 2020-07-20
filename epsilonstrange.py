import numpy as np
import epsilon_occ as ep
from sklearn import preprocessing
import pickle as pkl
# pickle_load = open("ssTableErrorEpsilon","rb")
# data = pkl.load(pickle_load)
# pickle_load.close()
data = np.load("ssTableErrorEpsilonnp.npy")
scaler = preprocessing.StandardScaler().fit(data)
clf = ep.RPEOCC(6,1e-2,0.9,1e-2)
clf.fit(scaler.transform(data))
initialpredictions = clf.predict(scaler.transform(data))
# print(len(data))
for i in range(len(data)):
	newpred = clf.predict(scaler.transform([data[i]]))
	if(newpred!=initialpredictions[i]):
		print(i, data[i], scaler.transform([data[i]]))
# print(data[8050])
# print(scaler.transform([data[8050]]).tolist())
# print(clf.predict(scaler.transform([data[8050]])))
# print(initialpredictions[8050])
