import util
import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import StratifiedKFold

'''
	CLASSIFICADOR I
	BAYESIANO GAUSSIANO
	NORMAL MULTIVARIADA
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
train_index, test_index = next(iter(skf.split(fac_trainingSet, target_training)))

X_train = fac_trainingSet[train_index]
y_train = target_training[train_index]
X_test = fac_trainingSet[test_index]
y_test = target_training[test_index]

n_classes = len(np.unique(y_train))

print("%s" % (n_classes))

# Try GMMs using different types of covariances.
estimator = GaussianMixture(n_components=n_classes, covariance_type='diag', max_iter=20, random_state=0)

# initialize the GMM parameters in a supervised manner.
estimator.means_init = np.array([X_train[y_train == i].mean(axis=0) for i in range(n_classes)])

# Train the other parameters using the EM algorithm.
estimator.fit(X_train)

y_train_pred = estimator.predict(X_train)
train_accuracy = np.mean(y_train_pred.ravel() == y_train.ravel()) * 100
print('Train accuracy: %.1f' % train_accuracy)

y_test_pred = estimator.predict(X_test)
test_accuracy = np.mean(y_test_pred.ravel() == y_test.ravel()) * 100
print('Test accuracy: %.1f' % test_accuracy)
