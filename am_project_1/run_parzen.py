import util
import parzen
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn import preprocessing
from sklearn.decomposition import PCA

'''
    RUN BAYESIAN PARZEN WINDOW CLASSIFIER
    3 VIEWS
    FOU - FAC- KAR
'''

# number of classes
numberOfClasses = 10
# patterns per class
patternSpace = 200
# number of repetitions
repetitions = 30

# path to datasets
fac_file = "mfeat/mfeat-fac"
fou_file = "mfeat/mfeat-fou"
kar_file = "mfeat/mfeat-kar"

# Get datasets as a numpy 2d array
fac = util.readDataset(fac_file)
fou = util.readDataset(fou_file)
kar = util.readDataset(kar_file)

fac = preprocessing.scale(fac)
fou = preprocessing.scale(fou)
kar = preprocessing.scale(kar)

# project the d-dimensional data to a lower dimension
pca = PCA(n_components=15, whiten=False)
fac = pca.fit_transform(fac)
fou = pca.fit_transform(fou)
kar = pca.fit_transform(kar)


# Generates numpy array of targets (classes)
target = util.generateTargets(numberOfClasses, patternSpace)

# stratified cross validation
rskf = RepeatedStratifiedKFold(n_splits=10, n_repeats=30, random_state=42)
#skf = StratifiedKFold(n_splits=10, random_state=42)


print("======================== FOU =============================")

#fou_h = parzen.bandwidth_estimator(fou)
fou_h = 1.9952
#print("fou best bandwidth: {0}".format(fou_h))

fou_predictions, fou_error_rates = parzen.runClassifier(rskf, fou, target, fou_h)

print("fou confusion matrix")
fouConfusionMatrix = util.confusionMatrix(fou_predictions)
print(np.array_str(fouConfusionMatrix, precision=6, suppress_small=True))
print("")

print("fou precision by class")
fou_precision_by_class = fouConfusionMatrix.diagonal() / float(patternSpace * repetitions)
print(fou_precision_by_class)
print("")

print("fou precision average %s" % np.mean(fou_precision_by_class))
print("fou error rate average %s" % util.errorRateAverage(fou_error_rates))
print("")

# erro rate for each repetition
for i, rate in enumerate(fou_error_rates):
    print ("repetition:", i, "error rate", rate)
print("")

print("======================== FAC =============================")
#fac_h = parzen.bandwidth_estimator(fac)
fac_h = 2.3865
#print("fac best bandwidth: {0}".format(fac_h))

fac_predictions, fac_error_rates = parzen.runClassifier(rskf, fac, target, fac_h)

print("fac confusion matrix")
facConfusionMatrix = util.confusionMatrix(fac_predictions)
print(np.array_str(facConfusionMatrix, precision=6, suppress_small=True))
print("")

print("fac precision by class")
fac_precision_by_class = facConfusionMatrix.diagonal() / float(patternSpace * repetitions)
print(fac_precision_by_class)
print("")

print("fac precision average %s" % np.mean(fac_precision_by_class))
print("fac error rate average %s" % util.errorRateAverage(fac_error_rates))
print("")

# erro rate for each repetition
for i, rate in enumerate(fac_error_rates):
    print ("repetition:", i, "error rate", rate)
print("")

print("======================== KAR =============================")
#kar_h = parzen.bandwidth_estimator(kar)
kar_h = 1.9952
#print("kar best bandwidth: {0}".format(kar_h))

kar_predictions, kar_error_rates = parzen.runClassifier(rskf, kar, target, kar_h)

print("kar confusion matrix")
karConfusionMatrix = util.confusionMatrix(kar_predictions)
print(np.array_str(karConfusionMatrix, precision=6, suppress_small=True))
print("")

print("kar precision by class")
kar_precision_by_class = karConfusionMatrix.diagonal() / float(patternSpace * repetitions)
print(kar_precision_by_class)
print("")

print("kar precision average %s" % np.mean(kar_precision_by_class))
print("kar error rate average %s" % util.errorRateAverage(kar_error_rates))
print("")

# erro rate for each repetition
for i, rate in enumerate(kar_error_rates):
    print ("repetition:", i, "error rate", rate)