import random
import numpy as np

'''
	READS DATASETS AND CONVERTS TO A NUMPY ARRAY
'''
def readDataset(dataset_name):
	aux = []
	with open(dataset_name) as file:
	    aux = [[float(value) for value in line.split()] for line in file]

	aux = np.array(aux)

	return aux
	
'''
	SPLITS THE DATASET INTO TRAINING AND TEST SETS
'''
def splitDataset(dataset, ratio):
	dataset = dataset.tolist()
	trainSet = []
	testSet = []
	for category in range(0, 10):
		rangeLimit = (category * 200) + 200
		#print("class: {0} | range limit: {1}").format(category, rangeLimit)
		categoryInterval = dataset[(category * 200) : rangeLimit]
		#print("Interval length: %s" % len(categoryInterval))
		IntervalLength = len(categoryInterval)
		trainSize = int(IntervalLength * ratio)
		#print("Train Size: %s" % trainSize)
		copy = list(categoryInterval)
		count = 0
		while count < trainSize:
			index = random.randrange(len(copy))
			trainSet.append(copy.pop(index))
			count = count + 1
		testSet.append(copy)
	testSet = [val for sublist in testSet for val in sublist]
	return np.array(trainSet), np.array(testSet)
