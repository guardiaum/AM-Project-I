import nayve_bayes as bayes
import numpy as np

'''
    CLASSIFICADOR I
    BAYESIANO GAUSSIANO
    VEROSSIMILHANCA
'''

# fit training set - estimates parameters
def computePriorsAndThetas(train_set, numberOfClasses):
    prior_ = []
    mu_ = []
    sigma_ = []

    # iterates through class training samples
    for class_w in range(0, numberOfClasses):

        # training set and class sample size
        train_sample_size = train_set.shape[0]
        class_sample_size = train_sample_size / 10

        # limits for training samples from class
        initial_sample = class_w * int(class_sample_size)
        end_sample = initial_sample + int(class_sample_size)

        # class samples
        w_samples = train_set[initial_sample:end_sample, :]

        # calculates prior for class
        prior = bayes.calculatePrior(class_sample_size, train_sample_size)

        # computes theta for each class
        mu, sigma = calculateTheta(w_samples)

        prior_.append(prior)
        mu_.append(mu)
        sigma_.append(sigma)

    return prior_, mu_, sigma_

# calculate theta from samples
def calculateTheta(samples):
    # calculates mean and standard deviation
    mu = np.mean(samples, axis=1)

    cov = []
    for i in range(0, samples.shape[0]):
        row_std = []
        for j in range(0, samples.shape[0]):
            std = 0
            if (i == j):
                std = np.std(samples[i, :])
            row_std.append(std)
        cov.append(row_std)

    sigma = np.array(cov)

    return mu, sigma

# calculate pdf for sample
def likelihood(x, mu, sigma):
    # sample size
    d = mu.shape[0]

    # computes probability density function for x given class
    first_statement = np.power(2 * np.pi, (-d/2))
    second_statement = np.power(np.product(sigma.diagonal()), -1/2)

    # for each sample calculates the natural exponential
    exp_statement = []
    for j in range(0, d):
        std = np.power(x - mu[j], 2) # standard deviation from each feature in sample
        exp = std / sigma[j, j] # sigma[j, j] > lambda
        exp_statement.append(exp)

    exp_statement = np.array(exp_statement)
    third_statement = np.exp(-1/2 * np.sum(exp_statement))
    #print(third_statement)

    # calculates pdf for sample
    likelihood = first_statement * second_statement * third_statement

    # pdfs from all training samples
    return np.array(likelihood)
