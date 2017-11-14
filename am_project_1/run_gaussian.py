import util
import nayve_bayes as bayes
import gaussian
import numpy as np
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import StratifiedKFold

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
rskf = RepeatedStratifiedKFold(n_splits=10, n_repeats=30, random_state=42)

#skf = StratifiedKFold(n_splits=10, random_state=42)

error_rates = []
predictions = []

r = 0
for train_index, test_index in rskf.split(fac, target):
    r += 1
    print("repetition %s" % r)

    train_set = fac[train_index]
    train_target = target[train_index]
    test_set = fac[test_index]
    test_target = target[test_index]

    # computes prior and theta for all classes
    prior_, mu_, sigma_ = gaussian.computePriorsAndThetas(train_set, numberOfClasses)

    # compute likelihood for each sample in test set
    true_positives = 0
    false_positives = 0
    repetition_predictions = []
    for i, test_sample in enumerate(test_set):

        likelihoods = []
        # iterates over each class
        # computing likelihood for each sample
        for w in range(0, numberOfClasses):
            # get class parameters
            class_mu = mu_[w]
            class_sigma = sigma_[w]

            # get test sample from test set
            x = test_sample

            # calculates likelihood for x given class w parameters
            likelihood = gaussian.likelihood(x, class_mu, class_sigma)
            likelihoods.append(likelihood)

            #print("class %s" % w)
            #print("pdf %s" % likelihood)

        likelihoods = np.array(likelihoods)

        # calcula evidencia de cada exemplo utilizando
        # as verossimilhancas do exemplo dado cada classe
        evidence = bayes.evidence(likelihoods, prior_)

        # computa posteriori para cada classe
        posteriors = []
        for w in range(0, numberOfClasses):
            posterior = bayes.posterior(prior_[w], likelihoods[w], evidence)
            posteriors.append(posterior)

        posteriors = np.array(posteriors)

        # afeta exemplo a classe de maior posteriori
        max = np.argmax(posteriors)
        #print("PREDICT: %s" % max)
        #print("ACTUAL CLASS: %s" % test_target[i])

        # usado para gerar matriz de confusao
        prediction = []
        prediction.append(test_target[i])
        prediction.append(max)
        repetition_predictions.append(prediction)

        # para calculo de taxa de erro
        if(test_target[i] == max):
            true_positives += 1
        else:
            false_positives +=1

    print("true positives: %s" % true_positives)
    print("false positives: %s" % false_positives)

    error_rate = util.errorRate(true_positives, false_positives)

    error_rates.append(error_rate)
    predictions.append(repetition_predictions)

print("confusion matrix")
print(util.confusionMatrix(predictions))
print("error rate average %s" % util.errorRateAverage(error_rates))