import numpy

'''
    NAYVE BAYES RULE
'''

# calculate evidence from all sample likelihoods
def evidence(likelihoods, priors):
    evidence = 0
    for i in range(0, likelihoods.shape[0]):
        evidence = evidence + (likelihoods[i] * priors[i])
    return evidence


# calculate the posterior probability of a sample given some class
def posterior(class_prior, likelihood, evidence):
    return (likelihood * class_prior) / evidence


# calculate prior for class
def calculatePrior(class_sample_size, train_sample_size):
    return class_sample_size / float(train_sample_size)