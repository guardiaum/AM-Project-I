import util
import nayve_bayes as bayes
import numpy as np

'''
    CLASSIFICADOR I
    BAYESIANO GAUSSIANO
    VEROSSIMILHANCA
'''

# number of classes
numberOfClasses = 10

# fit training set - estimates parameters
def estimateParameters(train_set, numberOfClasses):
    mu_ = []
    sigma_ = []

    # iterates through class training samples
    for class_w in range(0, numberOfClasses):

        # training set and class sample size
        train_sample_size = train_set.shape[0]
        class_sample_size = train_sample_size / numberOfClasses

        # limits for training samples from class
        initial_sample = class_w * int(class_sample_size)
        end_sample = initial_sample + int(class_sample_size)

        # class samples
        w_samples = train_set[initial_sample:end_sample, :]

        # computes theta for each class
        mu, sigma = calculateTheta(w_samples)

        mu_.append(mu)
        sigma_.append(sigma)

    return mu_, sigma_

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

# predict sample using estimated parameters
def predict(test_sample, mu_, sigma_, prior_):
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
        density = likelihood(x, class_mu, class_sigma)
        likelihoods.append(density)

        # print("class %s" % w)
        # print("pdf %s" % likelihood)

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
    return np.argmax(posteriors), posteriors[np.argmax(posteriors)]

# run classifier withh stratified cros validation
def runClassifier(rskf, dataset, target):
    error_rates = []
    predictions = []

    r = 0
    for train_index, test_index in rskf.split(dataset, target):
        r += 1
        #print("repetition %s" % r)

        train_set = dataset[train_index]
        test_set = dataset[test_index]
        test_target = target[test_index]

        # compute priors
        prior_ = bayes.calculatePrior(train_set, numberOfClasses)

        # computes theta for all classes
        mu_, sigma_ = estimateParameters(train_set, numberOfClasses)

        # predict class for each sample in test set
        true_positives = 0
        false_positives = 0
        samples_prediction = []
        for i, test_sample in enumerate(test_set):
            # true class for sample
            actual_class = test_target[i]

            # afeta exemplo a classe de maior posteriori
            predicted_class, posteriori = predict(test_sample, mu_, sigma_, prior_)

            # print("PREDICT: %s" % predicted_class)
            # print("ACTUAL CLASS: %s" % actual_class)

            # usado para gerar matriz de confusao
            prediction = []
            prediction.append(actual_class)
            prediction.append(predicted_class)
            samples_prediction.append(prediction)

            # para calculo de taxa de erro
            if (actual_class == predicted_class):
                true_positives += 1
            else:
                false_positives += 1

        #print("true positives: %s" % true_positives)
        #print("false positives: %s" % false_positives)

        error_rate = util.errorRate(true_positives, false_positives)

        error_rates.append(error_rate)
        predictions.append(samples_prediction)

    return predictions, error_rates