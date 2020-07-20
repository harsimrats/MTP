import pickle
import pprint
from pybloom import BloomFilter
import epsilon_occ as ep
import numpy as np
import math
from interface import *

columnFamily = {} #columnFamily['name'] = {} dict of sstables (= dictSStable['sstableName'] = {}, dictSStable['sstableName']['bf'], dictSStable['sstableName']['model'], dictSStable['sstableName']['sstables']})

fp = {} # sstablename : false positives
tp = {} # sstablename : true positive
fn = {}

use_bf = 1
use_model = 1

bf_error_rate = 0.1

params = {"D":6, "epsilon":1e-2, "agg_thr":0.9, "box_thr":1e-2}

def readFiles():
	f_flush = open('finalLogs/flush', 'r')
	f_read = open('finalLogs/read', 'r')
	f_compacted = open('finalLogs/compacted', 'r')

	flushlst = f_flush.readlines()
	readlst = f_read.readlines()
	compactedlst = f_compacted.readlines()

	datalst = []
	flushlst = [ele[1:-1].split(' ') for ele in flushlst]
	readlst = [ele[1:-1].split(' ') for ele in readlst]
	compactedlst = [ele[1:-1].split(' ') for ele in compactedlst]

	datalst = [flushlst, readlst, compactedlst]

	return datalst

def pickleData(columnFamily, fp):
	data = []
	data.append(columnFamily)
	data.append(fp)
	file = open('datalogs', 'wb')
	pickle.dump(data, file)
	file.close()

def checkCond(ind, datalst):
	if ind[0] >= len(datalst[0]) and ind[1] >= len(datalst[1]) and ind[2] >= len(datalst[2]):
		return True
	else:
		return False

def getIndexfromtime(time_read, time_flush, time_compacted):
	time = []
	time.append(time_flush.replace(',', '.'))
	time.append(time_read.replace(',', '.'))
	time.append(time_compacted.replace(',', '.'))
	x = time.index(min(time))
	return x

def updateTime(ind, datalst):
	if ind[0] >= len(datalst[0]):
		time_flush = '24:60:60'
	else:
		time_flush = datalst[0][ind[0]][3]
	
	if ind[1] >= len(datalst[1]):
		time_read = '24:60:60'
	else:
		time_read = datalst[1][ind[1]][3]
	
	if ind[2] >= len(datalst[2]):
		time_compacted = '24:60:60'
	else:
		time_compacted = datalst[2][ind[2]][3]

	return time_read, time_compacted, time_flush

def doflush(datalst, ind, currInd):
	global columnFamily, fp, bf_error_rate

	def removetmp(name):
		hInd = name.replace('-', '\\', 1).find('-')
		newName = name[:hInd] + name[hInd+4:]
		return newName

	sstableName = datalst[0][ind[currInd]][-1]
	# sstableName = removetmp(sstableName)

	try:
		columnFamilyName = sstableName.split('/')[-2]
	except:
		print(sstableName, ind[currInd])

	key = datalst[0][ind[currInd]][10][:-1]

	if key.startswith('DecoratedKey('):
		key = key[13:]

	# check and create column family
	if columnFamilyName not in columnFamily:
		columnFamily[columnFamilyName] = {}
	
	if columnFamilyName == 'medusa_vdiskblockmap':
		use_bf = 1
	else:
		use_bf = 0

	# check and create sstable
	if sstableName not in columnFamily[columnFamilyName]:
		columnFamily[columnFamilyName][sstableName] = {}
		columnFamily[columnFamilyName][sstableName]['sstables'] = []
		if use_bf == 1:
			columnFamily[columnFamilyName][sstableName]['bf'] = BloomFilter(capacity=10000, error_rate=bf_error_rate)
		
	# add key to sstable
	columnFamily[columnFamilyName][sstableName]['sstables'].append(key)
	if use_bf == 1:
		columnFamily[columnFamilyName][sstableName]['bf'].add(key)

def doread(datalst, ind, currInd):
	global columnFamily, fp, tp
	global numreads, numreadsimulated, numreadcolexist, numreadnotfound, numreadcolnotfound

	numreads += 1

	try:
		readkey = datalst[1][ind[currInd]][29][:-1].split('=')[1]
		columnFamilyName = datalst[1][ind[currInd]][28].split('(')[1].split('=')[1][1:-2]
	except:
		print(datalst[1][ind[currInd]], ind[currInd])
		exit()

	if columnFamilyName == 'medusa_vdiskblockmap':
		use_bf = 1
		use_model = 1
	else:
		use_bf = 0
		use_model = 0

	if columnFamilyName in columnFamily:
		numreadcolexist += 1

		# find sstable and update fps
		flag = 0
		ansSStable = ''
		for sstable in reversed(columnFamily[columnFamilyName].keys()):
			bf_condition = True
			if use_bf == 1:
				bf_condition = readkey in columnFamily[columnFamilyName][sstable]['bf']
			
			if bf_condition:
				for key in columnFamily[columnFamilyName][sstable]['sstables']:
					if readkey == key:
						numreadsimulated += 1
						flag = 1
						ansSStable = sstable
						if sstable not in tp:
							tp[sstable] = {}
							tp[sstable]['list'] = []
						tp[sstable]['list'].append(readkey)
						break

			if flag == 1:
				break

		if flag == 1:
			for sstable in reversed(columnFamily[columnFamilyName].keys()):
				bf_condition = True
				if use_bf == 1:
					bf_condition = readkey in columnFamily[columnFamilyName][sstable]['bf']

				if columnFamilyName == 'medusa_vdiskblockmap':
					if use_model:
						if 'model' not in columnFamily[columnFamilyName][sstable]:
							# create and train model
							columnFamily[columnFamilyName][sstable]['model'] = interface.getmodel(columnFamily[columnFamilyName][sstable]['sstables'], columnFamilyName)

					# model_condition = predict model on read key

					model_condition = interface.predict(readkey, columnFamily[columnFamilyName][sstable]['model'])

					mc = 1
					for  i  in range(0, len(model_condition)):
						if model_condition[i] == 0:
							mc = 0
							break
					
					if mc == 1:
						# update fp
						if sstable not in fp:
							fp[sstable] = {}
							fp[sstable]['list'] = []
							fp[sstable]['size'] = len(columnFamily[columnFamilyName][sstable]['sstables'])
						fp[sstable]['list'].append(readkey)
						fp[sstable]['size'] = len(columnFamily[columnFamilyName][sstable]['sstables'])
					else:
						if bf_condition:
							if sstable not in fn:
								fn[sstable] = {}
								fn[sstable]['list'] = []
								fn[sstable]['size'] = len(columnFamily[columnFamilyName][sstable]['sstables'])
							fn[sstable]['list'].append(readkey)
							fn[sstable]['size'] = len(columnFamily[columnFamilyName][sstable]['sstables'])

							if sstable == ansSStable:
								break
							# update fp
							if sstable not in fp:
								fp[sstable] = {}
								fp[sstable]['list'] = []
								fp[sstable]['size'] = len(columnFamily[columnFamilyName][sstable]['sstables'])
							fp[sstable]['list'].append(readkey)
							fp[sstable]['size'] = len(columnFamily[columnFamilyName][sstable]['sstables'])
		
		else:
			numreadnotfound += 1
			# print(columnFamilyName, readkey)
			pass

	else:
		numreadcolnotfound += 1
		# print(columnFamilyName, readkey)
		# exit()

def docompaction(datalst, ind, currInd):
	global columnFamily, fp, bf_error_rate
	global  numcomapctionssimulated, numcomapctions, numComCol, numComSS, numColB

	numcomapctions += 1
	namesInd = []
	oldSStables = []
	for i, s in enumerate(datalst[2][ind[currInd]]):
		if 'SSTableReader' in s:
			namesInd.append(i)
	for i in namesInd:
		s = datalst[2][ind[currInd]][i]
		oldSStables.append(s[s.find('/'):-2])

	newsstableName = oldSStables[-1]
	oldSStables = oldSStables[:-1]
	
	# tranfer data from old sstable to new sstable
	keys = []
	flag = 0
	fc = 0
	fs = 0
	for i in range(0, len(oldSStables)):
		columnFamilyName = oldSStables[i].split('/')
		columnFamilyName = columnFamilyName[-2]
		if columnFamilyName not in columnFamily:
			# print('--------------------- NOT IN COLUMN FAMILY')
			fc = 1
			flag = 1

		elif oldSStables[i] not in columnFamily[columnFamilyName]:
			# print('--------------------- SSTABLE not present')
			# print(oldSStables[i])
			fs = 1
			flag = 1

		else:
			for key in columnFamily[columnFamilyName][oldSStables[i]]['sstables']:
				keys.append(key)
			del columnFamily[columnFamilyName][oldSStables[i]]

	if columnFamilyName == 'medusa_vdiskblockmap':
		use_bf = 1
		use_model = 1
	else:
		use_bf = 1
		use_model = 0

	if flag == 0:
		numcomapctionssimulated += 1
		newColumnFamilyName = newsstableName.split('/')[-2]
		if newColumnFamilyName not in columnFamily:
			columnFamily[newColumnFamilyName] = {}
		columnFamily[newColumnFamilyName][newsstableName] = {}
		columnFamily[newColumnFamilyName][newsstableName]['sstables'] = []
		if use_bf == 1:
			columnFamily[newColumnFamilyName][newsstableName]['bf'] = BloomFilter(capacity=len(keys), error_rate=bf_error_rate)
		
		for key in keys:
			columnFamily[newColumnFamilyName][newsstableName]['sstables'].append(key)
			if use_bf == 1:
				columnFamily[newColumnFamilyName][newsstableName]['bf'].add(key)

		if use_model == 1:
			# create model
			columnFamily[newColumnFamilyName][newsstableName]['model'] = interface.getmodel(columnFamily[newColumnFamilyName][newsstableName]['sstables'], newColumnFamilyName)

	elif fs == 0 and fc == 1:
		numComCol += 1

	elif fs == 1 and fc == 0:
		numComSS += 1

	elif fs == 1 and fc == 1:
		numColB += 1

	# if ind[currInd] == 5:
	# 	print(newColumnFamilyName, newsstableName)
	# 	print(columnFamily['default_WAL_keyspace'])
	# 	exit()

numcomapctions = 0
numcomapctionssimulated = 0
numComSS = 0
numComCol = 0
numColB = 0

numreads = 0
numreadsimulated = 0
numreadcolexist = 0
numreadnotfound = 0
numreadcolnotfound = 0


datalst = readFiles() # flush, read, compacted
# print(len(datalst[0]), len(datalst[1]))
# 164531 1594624


# currInd = 0
# while ind[currInd] < len(datalst[0]):
# 	doflush(datalst, ind, currInd)
# 	ind[currInd] += 1

# currInd = 1
# while ind[currInd] < len(datalst[1]):
# 	doread(datalst, ind, currInd)
# 	ind[currInd] += 1

ind = [0, 0, 0]		# flush, read, compacted
# while True:
# 	if ind[0] >= len(datalst[0]):
# 		break
	
# 	# time_read, time_compacted, time_flush = updateTime(ind, datalst)
# 	# currInd = getIndexfromtime(time_read, time_flush, time_compacted)
# 	# print(ind)
# 	currInd = 0

# 	if currInd == 0:		# flush
# 		doflush(datalst, ind, currInd)

# 	# if currInd == 1:		#read
# 	# 	doread(datalst, ind, currInd)

# 	# if currInd == 2:		#compaction
# 		# docompaction(datalst, ind, currInd)

# 	ind[currInd] += 1


interface = Interface(lessData=True)

while True:
	if checkCond(ind, datalst):
		break
	
	time_read, time_compacted, time_flush = updateTime(ind, datalst)
	currInd = getIndexfromtime(time_read, time_flush, time_compacted)

	if currInd == 0:		# flush
		doflush(datalst, ind, currInd)

	if currInd == 1:		#read
		doread(datalst, ind, currInd)

	if currInd == 2:		#compaction
		docompaction(datalst, ind, currInd)
		
	ind[currInd] += 1


# print(numcomapctions, numcomapctionssimulated, numColB, numComSS, numComCol)
# 132 132 0 0 0 -- on testlogs3

# print(numreads, numreadsimulated, numreadcolexist, numreadnotfound, numreadcolnotfound)
# 1594624 461194 1594615 1133421  -- on final logs
# 1449748 1329899 1449388 119489 -- on test logs
# 1245347 642893 637152 14184 -- on testlogs3

# on 4 node 20TB
# 107 107 0 0 0
# 1098687 228745 324805 96060 773882



for n in fn.keys():
	# print(n, len(fp[n]['list']), fp[n]['size'])
	# if n in tp.keys():
	print(n, len(set(fn[n]['list']))/float(fn[n]['size']), len(fn[n]['list']), fn[n]['size'], len(set(fn[n]['list'])))

# for n in tp.keys():
# 	print(n, len(tp[n]['list']))

# pp = pprint.PrettyPrinter(indent=4)
# pp.pprint(columnFamily)



