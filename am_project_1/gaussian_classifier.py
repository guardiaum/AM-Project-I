import util
import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import StratifiedKFold

'''
	CLASSIFICADOR I
	BAYESIANO GAUSSIANO
	VEROSSIMILHANCA
'''

# number of classes
numberOfClasses = 10
# patterns per class
patternSpace = 200

# path to datasets
fac_file = "mfeat/mfeat-fac"
fou_file = "mfeat/mfeat-fou"
kar_file = "mfeat/mfeat-kar"

# Get datasets as a numpy 2d array
fac_trainingSet = util.readDataset(fac_file)
fou_trainingSet = util.readDataset(fou_file)
kar_trainingSet = util.readDataset(kar_file)

# Generates numpy array of targets (classes)
target_training = util.generateTargets(numberOfClasses, patternSpace)

# Break up the dataset into non-overlapping training (75%) and testing
# (25%) sets.
skf = StratifiedKFold(n_splits=4)

# Only take the first fold.
fac_train_index, fac_test_index = next(iter(skf.split(fac_trainingSet, target_training)))
fac_train = fac_trainingSet[fac_train_index]
fac_target_train = target_training[fac_train_index]
fac_test = fac_trainingSet[fac_test_index]
fac_target_test = target_training[fac_test_index]

# initiates gaussian classifiers
# GaussianNB implements the Gaussian Naive Bayes algorithm for classification. The likelihood of the features is assumed to be Gaussian. The parameters \sigma_y and \mu_y are estimated using maximum likelihood.
fac_clf = GaussianNB()
#fou_clf = GaussianNB()
#kar_clf = GaussianNB()

# trains models
fac_clf.fit(fac_train, fac_target_train)
#fou_clf.fit(fou_trainingSet, target_training)
#kar_clf.fit(kar_trainingSet, target_training)

# test models
y_train_pred = fac_clf.predict(fac_train)
train_accuracy = np.mean(y_train_pred.ravel() == fac_target_train.ravel()) * 100
print('Train accuracy: %.1f' % train_accuracy)

y_test_pred = fac_clf.predict(fac_test)
test_accuracy = np.mean(y_test_pred.ravel() == fac_target_test.ravel()) * 100
print('Test accuracy: %.1f' % test_accuracy)

# print class_prior
print("Sigma ++++++++++++++++++")
print("fac: %s" % fac_clf.class_prior_)
#print("fou: %s" % fou_clf.class_prior_)
#print("kar: %s" % kar_clf.class_prior_)

# print sigma - variance
print("Sigma ++++++++++++++++++")
print("fac: %s, %s" % fac_clf.sigma_.shape)
#print("fou: %s, %s" % fou_clf.sigma_.shape)
#print("kar: %s, %s" % kar_clf.sigma_.shape)

# print theta - mean
print("Theta ++++++++++++++++++")
print("fac: %s, %s" % fac_clf.theta_.shape)
#print("fou: %s, %s" % fou_clf.theta_.shape)
#print("kar: %s, %s" % kar_clf.theta_.shape)
