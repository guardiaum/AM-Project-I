import util
import gaussian
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import StratifiedKFold
from sklearn import preprocessing

'''
    RUN BAYESIAN GAUSSIAN CLASSIFIER
    3 VIEWS
    FOU - FAC - KAR
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

fac = preprocessing.scale(fac)
fou = preprocessing.scale(fou)
kar = preprocessing.scale(kar)

# Generates numpy array of targets (classes)
target = util.generateTargets(numberOfClasses, patternSpace)

# stratified cross validation
#rskf = RepeatedStratifiedKFold(n_splits=10, n_repeats=30, random_state=42)
rskf = StratifiedKFold(n_splits=10, random_state=42)

fou_predictions, fou_error_rates = gaussian.runClassifier(rskf, fou, target)

print("fou confusion matrix")
print(util.confusionMatrix(fou_predictions))
print("fou error rate average %s" % util.errorRateAverage(fou_error_rates))

fac_predictions, fac_error_rates = gaussian.runClassifier(rskf, fac, target)

print("fac confusion matrix")
print(util.confusionMatrix(fac_predictions))
print("fac error rate average %s" % util.errorRateAverage(fac_error_rates))

kar_predictions, kar_error_rates = gaussian.runClassifier(rskf, kar, target)

print("kar confusion matrix")
print(util.confusionMatrix(kar_predictions))
print("kar error rate average %s" % util.errorRateAverage(kar_error_rates))