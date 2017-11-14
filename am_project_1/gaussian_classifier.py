import util
import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import RepeatedStratifiedKFold

'''
	CLASSIFICADOR I
	BAYESIANO GAUSSIANO
	VEROSSIMILHANCA
'''

def fitGaussian(train, target):
    # GaussianNB implements the Gaussian Naive Bayes algorithm for classification.
    # The likelihood of the features is assumed to be Gaussian.
    # The parameters \sigma_y and \mu_y are estimated using maximum likelihood.
    clf = GaussianNB()

    # fit model
    return clf.fit(train, target)


def testModel(classifier, train, test, target_train, target_test):

    train_pred = classifier.predict(train)
    train_accuracy = np.mean(train_pred.ravel() == target_train.ravel()) * 100
    test_pred = classifier.predict(test)
    test_accuracy = np.mean(test_pred.ravel() == target_test.ravel()) * 100

    return train_accuracy, test_accuracy

def trainModel(rskf, dataset, target):
    print("training...")

    trains_accuracy = []
    tests_accuracy = []

    for train_index, test_index in rskf.split(dataset, target):
        # Subset training and test datasets
        train = dataset[train_index]
        target_train = target[train_index]
        test = dataset[test_index]
        target_test = target[test_index]

        # train model
        clf = fitGaussian(train, target_train)

        '''
        # print class_prior
        print("priors ++++++++++++++++++")
        print("%s" % clf.class_prior_)

        # print sigma - variance
        print("sigma ++++++++++++++++++")
        print("%s, %s" % clf.sigma_.shape)

        # print theta - mean
        print("theta ++++++++++++++++++")
        print("%s, %s" % clf.theta_.shape)
        '''

        # test model
        train_accuracy, test_accuracy = testModel(clf, train, test, target_train, target_test)
        trains_accuracy.append(train_accuracy)
        tests_accuracy.append(test_accuracy)

    return np.mean(trains_accuracy), np.mean(tests_accuracy)

# number of classes
numberOfClasses = 10
# patterns per class
patternSpace = 200

# path to datasets
fac_file = "mfeat/mfeat-fac"
fou_file = "mfeat/mfeat-fou"
kar_file = "mfeat/mfeat-kar"

# Get datasets as a numpy 2d array
fac = util.readDataset(fac_file)
fou = util.readDataset(fou_file)
kar = util.readDataset(kar_file)

# Generates numpy array of targets (classes)
target = util.generateTargets(numberOfClasses, patternSpace)

# stratified cross validation
rskf = RepeatedStratifiedKFold(n_splits=10, n_repeats=30, random_state=42)

# FAC VIEW
train_accuracy, test_accuracy = trainModel(rskf, fac, target)
print('fac view')
print('Train accuracy: %s' % train_accuracy)
print('Test accuracy: %s' % test_accuracy)

# FOU VIEW
train_accuracy, test_accuracy = trainModel(rskf, fou, target)
print('fou view')
print('Train accuracy: %s' % train_accuracy)
print('Test accuracy: %s' % test_accuracy)

# KAR VIEW
train_accuracy, test_accuracy = trainModel(rskf, kar, target)
print('kar view')
print('Train accuracy: %s' % train_accuracy)
print('Test accuracy: %s' % test_accuracy)