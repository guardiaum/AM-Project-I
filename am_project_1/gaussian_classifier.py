import util
from sklearn.naive_bayes import GaussianNB

# patterns per class
patternSpace = 200
# split ratio
ratio = 0.60

# path to datasets
fac_file = "mfeat/mfeat-fac"
fou_file = "mfeat/mfeat-fou"
kar_file = "mfeat/mfeat-kar"

# Get datasets as a numpy 2d array
fac = util.readDataset(fac_file)
fou = util.readDataset(fou_file)
kar = util.readDataset(kar_file)

# Generates numpy array of targets (classes)
target_training, target_test = util.generateTargets(patternSpace, ratio)
print('target training size: {0} | target test size: {1}').format(target_training.shape[0], target_test.shape[0])

# splits datasets into training and test sets
# ratio is dataset percentage to be used for training the rest is for testing
if(ratio != 1.0):
	fac_trainingSet, fac_testSet = util.splitDataset(fac, ratio)
	fou_trainingSet, fou_testSet = util.splitDataset(fou, ratio)
	kar_trainingSet, kar_testSet = util.splitDataset(kar, ratio)
	# print split output
	print('fac dataset - split {0} rows into train with {1} and test with {2}').format(fac.shape[0], fac_trainingSet.shape[0], fac_testSet.shape[0])
	print('fou dataset - split {0} rows into train with {1} and test with {2}').format(fou.shape[0], fou_trainingSet.shape[0], fou_testSet.shape[0])
	print('kar dataset - split {0} rows into train with {1} and test with {2}').format(kar.shape[0], kar_trainingSet.shape[0], kar_testSet.shape[0])


# initiates gaussian classifiers
# GaussianNB implements the Gaussian Naive Bayes algorithm for classification. The likelihood of the features is assumed to be Gaussian. The parameters \sigma_y and \mu_y are estimated using maximum likelihood.
fac_clf = GaussianNB()
fou_clf = GaussianNB()
kar_clf = GaussianNB()

# trains models
fac_clf.fit(fac_trainingSet, target_training)
fou_clf.fit(fou_trainingSet, target_training)
kar_clf.fit(kar_trainingSet, target_training)

# test models
fac_test = fac_clf.predict(fac_testSet)
fou_test = fou_clf.predict(fou_testSet)
kar_test = kar_clf.predict(kar_testSet)

wrongPredictions = (target_test!=fac_test).sum()

# print tests
print("Number of mislabeled points out of a total %d points : %d" % (fac_trainingSet.shape[0], wrongPredictions))

# print sigma - variance
print("Sigma ++++++++++++++++++")
print("fac: %s, %s" % fac_clf.sigma_.shape)
print("fou: %s, %s" % fou_clf.sigma_.shape)
print("kar: %s, %s" % kar_clf.sigma_.shape)

# print theta - mean
print("Theta ++++++++++++++++++")
print("fac: %s, %s" % fac_clf.theta_.shape)
print("fou: %s, %s" % fou_clf.theta_.shape)
print("kar: %s, %s" % kar_clf.theta_.shape)
