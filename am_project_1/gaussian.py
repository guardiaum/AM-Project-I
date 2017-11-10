import util
import numpy as np
from scipy.stats import norm
from sklearn.model_selection import RepeatedStratifiedKFold

'''
    CLASSIFICADOR I
    BAYESIANO GAUSSIANO
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
rskf = RepeatedStratifiedKFold(n_splits=10, n_repeats=30, random_state=42)

for fac_train_index, fac_test_index in rskf.split(fac, target):

    print("FAC - TRAIN:", fac_train_index.shape, "TEST:", fac_test_index.shape)
    print("fac_features_train: (%s, %s)" % fac[fac_train_index].shape)
    print("fac_target_train: %s" % target[fac_train_index].shape)
    print("fac_features_test: (%s, %s)" % fac[fac_test_index].shape)
    print("fac_target_test: %s" % target[fac_test_index].shape)

    #iterates through class samples
    for class_w in range(0, 10):
        # limits for training samples from class
        initial_sample = class_w * (fac[fac_train_index].shape[0] / 10)
        end_sample = initial_sample + (fac[fac_train_index].shape[0] / 10)

        print("Class: %s" % class_w)
        #print("Start: %s" % initial_sample)
        #print("End: %s" % end_sample)

        logsLikelihood = []
        mus = []
        stds = []
        for feature_index in range(0, fac[fac_train_index].shape[1]):

            # get feature values from training samples
            w_samples = fac[initial_sample: end_sample, feature_index]

            # calculates mean and standard deviation
            mu = np.mean(w_samples)
            std = np.std(w_samples)

            # gets log of the probability density function.
            # log-likelihood
            log_likelihood = norm.logpdf(w_samples, loc=mu, scale=std)

            #print("")
            #print("mu: %s | std: %s" % (mu, std))
            #print(feature_index, log_likelihood.shape)

            logsLikelihood.append(log_likelihood)
            mus.append(mu)
            stds.append(std)

        
        logLikelihoodFunction = np.sum(logsLikelihood, axis=0)

        # get log-likelihood argmax for theta
        maxLikelihoodIndex = np.argmax(logLikelihoodFunction)
        max_loglikelihood = logLikelihoodFunction[maxLikelihoodIndex]
        max_theta = [mus[maxLikelihoodIndex], stds[maxLikelihoodIndex]]

        print("Max likelihood: ", max_loglikelihood)
        print("Max theta: %s" % max_theta)

    break