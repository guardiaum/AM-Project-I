import util
from sklearn.naive_bayes import GaussianNB

# split ratio
ratio = 0.60
patternSpace = 200

# path to datasets
fac_file = "mfeat/mfeat-fac"
fou_file = "mfeat/mfeat-fou"
kar_file = "mfeat/mfeat-kar"

# Get datasets as a numpy 2d array
fac = util.readDataset(fac_file)
fou = util.readDataset(fou_file)
kar = util.readDataset(kar_file)

# Generates array of targets (classes)
target = []
patternSpace = int(patternSpace * ratio)
for i in range(0,  10):
	target.append([i] * patternSpace)
target = [val for sublist in target for val in sublist]

# splits datasets into training and test sets
# 0.60 for training ad 0.30 for testing
fac_trainingSet, fac_testSet = util.splitDataset(fac, ratio)
fou_trainingSet, fou_testSet = util.splitDataset(fou, ratio)
kar_trainingSet, kar_testSet = util.splitDataset(kar, ratio)

# print split output
print('fac dataset - split {0} rows into train with {1} and test with {2}').format(fac.shape[0], fac_trainingSet.shape[0], fac_testSet.shape[0])
print('fou dataset - split {0} rows into train with {1} and test with {2}').format(fou.shape[0], fou_trainingSet.shape[0], fou_testSet.shape[0])
print('kar dataset - split {0} rows into train with {1} and test with {2}').format(kar.shape[0], kar_trainingSet.shape[0], kar_testSet.shape[0])

# initiates gaussian classifiers
fac_clf = GaussianNB()
fou_clf = GaussianNB()
kar_clf = GaussianNB()

# trains models
fac_clf.fit(fac_trainingSet, target)
fou_clf.fit(fou_trainingSet, target)
kar_clf.fit(kar_trainingSet, target)

# test models
fac_test = fac_clf.predict(fac_testSet)
fou_test = fou_clf.predict(fou_testSet)
kar_test = kar_clf.predict(kar_testSet)

# print tests
print(fac_test)
print(fou_test)
print(kar_test)
