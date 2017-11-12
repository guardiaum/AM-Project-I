import util
import numpy as np
from sklearn.neighbors.kde import KernelDensity
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold

'''
	CLASSIFICADOR I
	JANELA DE PARZEN
	KERNEL MULTIVARIADA DO PRODUTO
'''

# fits a better estimation for parameter h
def bandwidth_estimator(data):
    # use grid search cross-validation to optimize the bandwidth
    params = {'bandwidth': np.logspace(0.1, 1, 30)}
    grid = GridSearchCV(KernelDensity(), params, cv=30)
    grid.fit(data)

    return grid.best_estimator_.bandwidth


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
skf = StratifiedKFold(n_splits=10, random_state=42)

# iterates through class samples
for class_w in range(0, 10):
    print("Class: %s" % class_w)

    # limits for training samples from class
    initial_sample = class_w * (fac.shape[0] / 10)
    end_sample = initial_sample + (fac.shape[0] / 10)

    # get values from training samples
    fac_samples = fac[initial_sample: end_sample, :]
    target_samples = target[initial_sample: end_sample]

    h = bandwidth_estimator(fac_samples)

    print("best bandwidth: {0}".format(h))

    # Initiates Parzen classifiers
    fac_kde = KernelDensity(kernel='gaussian', bandwidth=h).fit(fac_samples)
    density = np.exp(fac_kde.score_samples(fac_samples))
    print(density)
