import numpy as np
import util
import nayve_bayes as bayes
import parzen
import gaussian
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn import preprocessing
from sklearn.decomposition import PCA

'''
    RUN BAYESIAN PARZEN WINDOW CLASSIFIER
    3 VIEWS
    FOU - FAC- KAR
'''

def ensemblePrediction(priors, posteriors):
    sub = 1 - 3
    multip = sub * priors
    sum_rule = multip + posteriors
    sum_rule = np.sum(sum_rule, axis=1)
    return np.argmax(sum_rule)

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

'''FIXED BANDWIDTHS'''
fou_h = 1.9952
fac_h = 2.3865
kar_h = 1.9952

print("fou best bandwidth: {0}".format(fou_h))
print("fac best bandwidth: {0}".format(fac_h))
print("kar best bandwidth: {0}".format(kar_h))

r = 0
error_rates = []
predictions = []
for train_index, test_index in rskf.split(fou, target):

    r += 1
    #print("repetition %s" % r)

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

    '''UNCOMMENT FOR NEW BANDWIDTH ESTIMATIONS'''
    # fou_h = parzen.bandwidth_estimator(fou_train_set)
    #print("fou best bandwidth: {0}".format(fou_h))
    # fac_h = parzen.bandwidth_estimator(fac_train_set)
    #print("fac best bandwidth: {0}".format(fac_h))
    # kar_h = parzen.bandwidth_estimator(kar_train_set)
    #print("kar best bandwidth: {0}".format(kar_h))

    # compute priors - same for all datasets
    prior_ = bayes.calculatePrior(fou_train_set, numberOfClasses)

    # computes theta for all classes
    fou_mu_, fou_sigma_ = gaussian.estimateParameters(fou_train_set, numberOfClasses)
    fac_mu_, fac_sigma_ = gaussian.estimateParameters(fac_train_set, numberOfClasses)
    kar_mu_, kar_sigma_ = gaussian.estimateParameters(kar_train_set, numberOfClasses)

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
        fac_parzen_posteriors = parzen.posteriorFromEachClass(fac_train_set, train_class_size, fac_test_sample,fac_h, prior_ )

        # calculate posteriors from vew kar using parzen window
        kar_parzen_posteriors = parzen.posteriorFromEachClass(kar_train_set, train_class_size, kar_test_sample,kar_h, prior_ )

        posteriors = zip(fou_gauss_posteriors, fac_gauss_posteriors, kar_gauss_posteriors,
                         fou_parzen_posteriors, fac_parzen_posteriors, kar_parzen_posteriors)

        # Ensemble classifiers using the sum rule
        predicted_class = ensemblePrediction(prior_, posteriors)

        #print("actual:", actual_class, " prediction:", predicted_class)
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
confusionMatrix = util.confusionMatrix(predictions)
print(np.array_str(confusionMatrix, precision=6, suppress_small=True))
print("")

print("precision by class")
precision_by_class = confusionMatrix.diagonal() / float(patternSpace * repetitions)
print(precision_by_class)
print("")

print("precision average %s" % np.mean(precision_by_class))
print("error rate average %s" % util.errorRateAverage(error_rates))
print("")

# erro rate for each repetition
for i, rate in enumerate(error_rates):
    print ("repetition:", i, "error rate", rate)
print("")
