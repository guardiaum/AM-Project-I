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


# calculate prior for classes
def calculatePrior(train_set, numberOfClasses):
    prior_ = []

    # iterates through class training samples
    for w in range(0, numberOfClasses):

        # training set and class sample size
        train_sample_size = train_set.shape[0]
        class_sample_size = train_sample_size / numberOfClasses

        #compute prior
        prior = class_sample_size / float(train_sample_size)
        prior_.append(prior)

    return prior_