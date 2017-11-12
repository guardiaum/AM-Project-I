import util
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
#rskf = RepeatedStratifiedKFold(n_splits=10, n_repeats=30, random_state=42)

skf = StratifiedKFold(n_splits=10, random_state=42)

for train_index, test_index in skf.split(fac, target):

    train_set = fac[train_index]
    train_target = target[train_index]
    test_set = fac[test_index]
    test_target = target[test_index]

    # computes prior and theta for all classes
    prior_, mu_, sigma_ = gaussian.computePriorsAndThetas(train_set, numberOfClasses)

    # compute likelihood for each sample in test set
    for i in range(0, test_set.shape[0]):

        likelihoods = []
        # iterates over each class
        # computing likelihood for each sample
        for w in range(0, numberOfClasses):

            class_mu = mu_[w]
            class_sigma = sigma_[w]

            x = test_set[i, :]

            # training set and class sample size
            test_sample_size = test_set.shape[0]
            class_sample_size = test_sample_size / 10

            # limits for training samples from class
            initial_sample = w * int(class_sample_size)
            end_sample = initial_sample + int(class_sample_size)

            # calculates likelihood for x given class w parameters
            likelihood = gaussian.likelihood(x, class_mu, class_sigma)
            likelihoods.append(likelihood)

            print("class %s" % w)
            #print("pdf %s" % likelihood)

        # calcula evidencia de cada exemplo utilizando
        # as verossimilhancas do exemplo dado cada classe
        likelihoods = np.array(likelihoods)
        evidence = gaussian.evidence(likelihoods, prior_)

        # computa posteriori para cada classe
        posteriors = []
        for w in range(0, numberOfClasses):
            posterior = gaussian.posterior(prior_[w], likelihoods[w], evidence)
            posteriors.append(posterior)

        posteriors = np.array(posteriors)

        # afeta exemplo a classe de maior posteriori
        max = np.argmax(posteriors)
        print("PREDICT: %s" % max)
        print("ACTUAL CLASS: %s" % test_target[i])

        if(test_target[i] == max):
            print("TRUE POSITIVE PREDICTION")
        else:
            print("FALSE POSITIVE PREDICTION")

        print("-----------------------------------------")