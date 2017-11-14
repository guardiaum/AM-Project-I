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
    return np.argmax(((1 - len(priors)) * priors) + posteriors)

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
error_rates = []
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
    test_sample_size = fou_test_set.shape[0]
    train_class_size = train_sample_size / numberOfClasses

    # compute priors - same for all datasets
    prior_ = bayes.calculatePrior(fou_train_set, numberOfClasses)

    # computes theta for all classes
    fou_mu_, fou_sigma_ = gaussian.estimateParameters(fou_train_set, numberOfClasses)
    fac_mu_, fac_sigma_ = gaussian.estimateParameters(fac_train_set, numberOfClasses)
    kar_mu_, kar_sigma_ = gaussian.estimateParameters(kar_train_set, numberOfClasses)

    # h = bandwidth_estimator(train_set)
    fou_h = 2 # fazer a estimativa de h para cada view
    fac_h = 2
    kar_h = 2

    # predict class for each sample in test set
    true_positives = 0
    false_positives = 0
    samples_predictions = []
    for sample_index in range(0, test_sample_size):

        fou_test_sample = fou_test_set[sample_index]
        fac_test_sample = fac_test_set[sample_index]
        kar_test_sample = kar_test_set[sample_index]

        # true class for sample - is the same true class for all views
        actual_class = fou_test_target[sample_index]

        # afeta exemplo a classe de maior posteriori
        # gaussian classifier
        # calculate posteriors from view fou using gaussian classifier
        fou_gauss_posteriors = gaussian.posteriorFromEachClass(fou_test_sample, fou_mu_, fou_sigma_, prior_)

        # calculate posteriors from view fac using gaussian classifier
        fac_gauss_posteriors = gaussian.posteriorFromEachClass(fac_test_sample, fac_mu_, fac_sigma_, prior_)

        # calculate posteriors from view kar using gaussian classifier
        kar_gauss_posteriors = gaussian.posteriorFromEachClass(kar_test_sample, kar_mu_, kar_sigma_, prior_)

        # calculate posteriors from view fou using parzen window
        fou_parzen_posteriors = parzen.posteriorFromEachClass(fou_train_set, train_class_size, fou_test_sample, fou_h, prior_)

        # calculate posteriors from view fac using parzen window
        fac_parzen_posteriors = parzen.posteriorFromEachClass(fac_train_set, train_class_size, fac_test_sample, fac_h, prior_)

        # calculate posteriors from vew kar using parzen window
        kar_parzen_posteriors = parzen.posteriorFromEachClass(kar_train_set, train_class_size, kar_test_sample, kar_h, prior_)

        posteriors = zip(fou_gauss_posteriors, fac_gauss_posteriors, kar_gauss_posteriors,
                         fou_parzen_posteriors, fac_parzen_posteriors, kar_parzen_posteriors)

        # Ensemble classifiers using the sum rule
        predicted_class = ensemblePrediction(prior_, posteriors)

        #print("true class: %s" % actual_class)
        #print("predicted: %s"% predicted_class)

        # usado para gerar matriz de confusao
        prediction = []
        prediction.append(actual_class)
        prediction.append(predicted_class)
        samples_predictions.append(prediction)

        # para calculo de taxa de erro
        if (actual_class == predicted_class):
            true_positives += 1
        else:
            false_positives += 1

    error_rate = util.errorRate(true_positives, false_positives)

    error_rates.append(error_rate)

    predictions.append(samples_predictions)

print("confusion matrix")
print(util.confusionMatrix(predictions))
print("error rate average %s" % util.errorRateAverage(error_rates))