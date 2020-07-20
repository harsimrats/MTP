import numpy as np
import epsilon_occ as ep
from sklearn import preprocessing
data = np.random.random((100,2))
scaler = preprocessing.StandardScaler().fit(data)
clf = ep.RPEOCC(6,1e-2,0.9,1e-2)
clf.fit(scaler.transform(data))
initialpredictions = clf.predict(scaler.transform(data))
for i in range(100):
	newpred = clf.predict(scaler.transform([data[i]]))
	if(newpred!=initialpredictions[i]):
		print(i, data[i], scaler.transform([data[i]]))