import random
import numpy as np

'''
	READS DATASETS AND CONVERTS TO A NUMPY ARRAY
'''


def readDataset(dataset_name):
    aux = []
    with open(dataset_name) as file:
        aux = [[float(value) for value in line.split()] for line in file]

    return np.array(aux)


'''
	GENERATES NUMPY ARRAY OF TARGETS (CLASSES) FOR PATTERN SPACE
'''


def generateTargets(numberOfClasses, patternSpace):
    target_train = []
    for i in range(0, numberOfClasses):
        target_train.append([i] * patternSpace)

    return np.hstack(target_train)


'''
	GENERATES NUMPY ARRAY OF TARGETS (CLASSES) FOR PATTERN SPACE CONSIDERING RATIO
'''


def generateSplittedTargets(numberOfClasses, patternSpace, ratio):
    target_train = []
    newPatternSpace = int(patternSpace * ratio)
    for i in range(0, numberOfClasses):
        target_train.append([i] * newPatternSpace)

    target_test = []
    ratio_test = 1.00 - ratio
    newPatternSpace = int(patternSpace * float(ratio_test))
    for i in range(0, numberOfClasses):
        target_test.append([i] * newPatternSpace)

    return np.hstack(target_train), np.hstack(target_test)


'''
	SPLITS THE DATASET INTO TRAINING AND TEST SETS
'''


def splitDataset(dataset, numberOfClasses, patternSpace, ratio):
    trainSet = []
    testSet = []
    for category in range(0, numberOfClasses):
        rangeLimit = (category * patternSpace) + patternSpace
        categoryInterval = list(dataset[(category * patternSpace): rangeLimit])
        IntervalLength = len(categoryInterval)
        trainSize = int(IntervalLength * ratio)
        copy = categoryInterval
        count = 0
        while count < trainSize:
            index = random.randrange(len(copy))
            trainSet.append(copy.pop(index))
            count = count + 1
        while len(copy) > 0:
            testSet.append(copy.pop())
    return np.asarray(trainSet, dtype=dataset.dtype), np.asarray(testSet, dtype=dataset.dtype)


'''
	CALCULATES ERROR RATE
'''


def errorRate(truepositives, falsepositives):
    return float(falsepositives) / (float(truepositives) + float(falsepositives))


'''
	CALCULATES ERROR RATE AVERAGE
'''


def errorRateAverage(error_rates):
    return np.mean(error_rates)


'''
	RETURN CONFUSION MATRIX GIVEN PREDICTIONS
'''


def confusionMatrix(repetitions_predictions):
    matrix = np.zeros((10, 10))
    for predictions in repetitions_predictions:
        for prediction in predictions:
            matrix[prediction[0], prediction[1]] += 1
    return matrix
