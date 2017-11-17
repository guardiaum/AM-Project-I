import util
import gaussian
import numpy as np
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import StratifiedKFold
from sklearn import preprocessing
from sklearn.decomposition import PCA


'''
    RUN BAYESIAN GAUSSIAN CLASSIFIER
    3 VIEWS
    FOU - FAC - KAR
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

fou_predictions, fou_error_rates = gaussian.runClassifier(rskf, fou, target)

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
print("error rates")
for i, rate in enumerate(fou_error_rates):
    print (rate)
print("")

# precision for each repetition
print("precision")
for i, rate in enumerate(fou_error_rates):
    print (1 - rate)
print("")


print("======================== FAC =============================")
fac_predictions, fac_error_rates = gaussian.runClassifier(rskf, fac, target)

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
print("error rates")
for i, rate in enumerate(fac_error_rates):
    print (rate)
print("")

# precision for each repetition
print("precision")
for i, rate in enumerate(fac_error_rates):
    print (1 - rate)
print("")

print("======================== KAR =============================")
kar_predictions, kar_error_rates = gaussian.runClassifier(rskf, kar, target)

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
print("error rates")
for i, rate in enumerate(kar_error_rates):
    print (rate)
print("")

# precision for each repetition
print("precision")
for i, rate in enumerate(kar_error_rates):
    print (1 - rate)
print("")