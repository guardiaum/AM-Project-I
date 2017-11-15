import util
import parzen
from sklearn.model_selection import StratifiedKFold

'''
    RUN BAYESIAN PARZEN WINDOW CLASSIFIER
    3 VIEWS
    FOU - FAC- KAR
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
fac = util.readDataset(fac_file)
fou = util.readDataset(fou_file)
kar = util.readDataset(kar_file)

# Generates numpy array of targets (classes)
target = util.generateTargets(numberOfClasses, patternSpace)

# stratified cross validation
# rskf = RepeatedStratifiedKFold(n_splits=10, n_repeats=30, random_state=42)
skf = StratifiedKFold(n_splits=10, random_state=42)

fou_h = parzen.bandwidth_estimator(fou)
print("fou best bandwidth: {0}".format(fou_h))

fou_predictions, fou_error_rates = parzen.runClassifier(skf, fou, target, fou_h)

print("fou confusion matrix")
print(util.confusionMatrix(fou_predictions))
print("error rate average %s" % util.errorRateAverage(fou_error_rates))

fac_h = parzen.bandwidth_estimator(fac)
print("fac best bandwidth: {0}".format(fac_h))

fac_predictions, fac_error_rates = parzen.runClassifier(skf, fac, target, fac_h)

print("fac confusion matrix")
print(util.confusionMatrix(fac_predictions))
print("fac error rate average %s" % util.errorRateAverage(fac_error_rates))


kar_h = parzen.bandwidth_estimator(kar)
print("fac best bandwidth: {0}".format(fac_h))

kar_predictions, kar_error_rates = parzen.runClassifier(skf, kar, target, kar_h)

print("kar confusion matrix")
print(util.confusionMatrix(kar_predictions))
print("kar error rate average %s" % util.errorRateAverage(kar_error_rates))