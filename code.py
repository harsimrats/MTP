import numpy as np
import math
from matplotlib import pyplot as plt

file = open('out.txt', 'r')

line = file.readlines()

lst = [ele[1:-2].split(' ') for ele in line]

data = []
minnum = float(lst[0][1])
maxnum = float(lst[0][1])
for ele in lst:
	data.append(float(ele[1]))
	minnum = min(minnum, float(ele[1]))
	maxnum = max(maxnum, float(ele[1]))

print(len(lst))
print(minnum, maxnum)


index = []
for i in range(len(data)):
	index.append(1+i)
plt.bar(index, data)
plt.xlabel('SSTables')
plt.ylabel('Ratio false positives to true positives that are queried but not in sstable')
plt.xticks(data, index, fontsize=5, rotation=30)
plt.title('Ratio fp to tp with error rate 0.9')
plt.show()

# bins = np.linspace(math.ceil(minnum), math.floor(maxnum), len(data)) # fixed number of bins
# # plt.xlim([minnum-5, maxnum+5])
# plt.hist(data, bins=bins, alpha=0.5)
# # plt.title('Random Gaussian data (fixed number of bins)')
# plt.ylabel('Ratio false positives to true positives that are queried but not in sstable')
# plt.xlabel('SSTables')
# plt.show()