import numpy as np
import util
import nayve_bayes as bayes
import parzen
import gaussian
from sklearn.model_selection import StratifiedKFold

'''
    RUN BAYESIAN PARZEN WINDOW CLASSIFIER
    3 VIEWS
    FOU - FAC- KAR
'''

def ensemblePrediction(priors, posteriors):
    return np.argmax(((1 - priors.shape[0]) * priors) + posteriors)

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

r = 0
predictions = []
for train_index, test_index in skf.split(fou, target):

    r += 1
    # print("repetition %s" % r)

    # datasets
    fou_train_set = fou[train_index]
    fou_test_set = fou[test_index]
    fou_test_target = target[test_index]

    fac_train_set = fac[train_index]
    fac_test_set = fac[test_index]
    fac_test_target = target[test_index]

    kar_train_set = kar[train_index]
    kar_test_set = kar[test_index]
    kar_test_target = target[test_index]

    # training set and class sample size - same for all datasets
    train_sample_size = fou_train_set.shape[0]
    train_class_size = train_sample_size / numberOfClasses

    # compute priors - same for all datasets
    prior_ = bayes.calculatePrior(fou_train_set, numberOfClasses)

    # computes theta for all classes
    fou_mu_, fou_sigma_ = gaussian.estimateParameters(fou_train_set, numberOfClasses)
    fac_mu_, fac_sigma_ = gaussian.estimateParameters(fac_train_set, numberOfClasses)
    kar_mu_, kar_sigma_ = gaussian.estimateParameters(kar_train_set, numberOfClasses)

    # h = bandwidth_estimator(train_set)
    fou_h = 2
    fac_h = 2
    kar_h = 2

    # predict class for each sample in test set
    samples_predictions = []
    for i, test_sample in enumerate(fou_test_set): # corrigir... utilizar as outras views

        # true class for sample
        actual_class = fou_test_target[i]

        # afeta exemplo a classe de maior posteriori
        # gaussian classifier
        fou_gauss_posteriors = gaussian.posteriorFromEachClass(test_sample, fou_mu_, fou_sigma_, prior_)

        fac_gauss_posteriors = gaussian.posteriorFromEachClass(test_sample, fac_mu_, fac_sigma_, prior_)

        kar_gauss_posteriors = gaussian.posteriorFromEachClass(test_sample, kar_mu_, kar_sigma_, prior_)

        # parzen window
        fou_parzen_posteriors = parzen.posteriorFromEachClass(fou_train_set, train_class_size, test_sample, fou_h, prior_)

        fac_parzen_posteriors = parzen.posteriorFromEachClass(fac_train_set, train_class_size, test_sample, fac_h, prior_)

        kar_parzen_posteriors = parzen.posteriorFromEachClass(kar_train_set, train_class_size, test_sample, kar_h, prior_)

        posteriors = zip(fou_gauss_posteriors, fac_gauss_posteriors, kar_gauss_posteriors,
                         fou_parzen_posteriors, fac_parzen_posteriors, kar_parzen_posteriors)

        predicted = ensemblePrediction(prior_, posteriors)

        # usado para gerar matriz de confusao
        prediction = []
        prediction.append(actual_class)
        prediction.append(predicted)
        samples_predictions.append(prediction)

    predictions.append(samples_predictions)