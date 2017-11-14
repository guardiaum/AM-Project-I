import util
import nayve_bayes as bayes
import numpy as np
from sklearn.neighbors.kde import KernelDensity
from sklearn.model_selection import GridSearchCV

'''
	CLASSIFICADOR I
	BAYESIANO JANELA DE PARZEN
	FUNCAO KERNEL MULTIVARIADA DO PRODUTO
'''

# number of classes
numberOfClasses = 10

# multivariate kernel function
# h = bandwidth
# x = point for dennsity estimation
# x_i = samples
def kernel_function(h, x, x_i):
    return (x - x_i) / h


# product from kernel function
def gaussian_window_func(phi):
    return np.product(phi)


# kernel estimator
# x_samples = training samples
# point_x = test sample
# h = bandwidth
# d = number of dimensions
def parzen_estimation(x_samples, point_x, h, d, kernel_func, window_func):
    n = x_samples.shape[0]
    v = h**d
    k = 0

    for sample in x_samples:
        phi = kernel_func(h, point_x, sample)
        k += window_func(phi)

    density = (1/n) * (1/float(v)) * float(k)

    #print("k", k)
    #print("n", n)
    #print("v", v)
    #print("density", density)

    return density


# fits a better estimation for parameter h
def bandwidth_estimator(data):
    # use grid search cross-validation to optimize the bandwidth
    params = {'bandwidth': np.logspace(0.2, 1, 20)}
    grid = GridSearchCV(KernelDensity(), params, cv=5)
    grid.fit(data)
    return grid.best_estimator_.bandwidth

def posteriorFromEachClass(train_set, train_class_size, test_sample, h, prior_):
    # iterates through classes calculating the density
    # for x given the class w
    densities = []
    for w in range(0, numberOfClasses):
        # limits for training samples from class
        train_initial_sample = w * int(train_class_size)
        train_end_sample = train_initial_sample + int(train_class_size)
        train_samples = train_set[train_initial_sample:train_end_sample, :]

        # calculates density through parzen window estimation with gaussian kernel
        density = parzen_estimation(train_samples, test_sample, h=h, d=train_samples.shape[1],
                                    kernel_func=kernel_function,
                                    window_func=gaussian_window_func)

        densities.append(density)
    densities = np.array(densities)

    # calculate the evidence
    evidence = bayes.evidence(densities, prior_)

    # calculates posteriors
    # w given x
    posteriors = []
    for w in range(0, numberOfClasses):
        posterior = bayes.posterior(prior_[w], densities[w], evidence)
        posteriors.append(posterior)
        # print("prior", prior_[w], "density", densities[w], "evidence", evidence)
        # print("w", w, "posterior", posterior)

    return np.array(posteriors)

# predict sample using estimated h parameter
def predict(train_set, train_class_size, test_sample, h, prior_):

    posteriors = posteriorFromEachClass(train_set, train_class_size, test_sample, h, prior_)

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

        # h = bandwidth_estimator(train_set)
        h = 2
        #print("best bandwidth: {0}".format(h))

        # training set and class sample size
        train_sample_size = train_set.shape[0]
        train_class_size = train_sample_size / numberOfClasses

        true_positives = 0
        false_positives = 0
        repetition_predictions = []

        # calculates prior probability for classes
        prior_ = bayes.calculatePrior(train_set, numberOfClasses)

        # calculates density for each test sample
        for i, test_sample in enumerate(test_set):

            actual_class = test_target[i]

            predicted_class, posteriori = predict(train_set, train_class_size, test_sample, h, prior_)

            # usado para gerar matriz de confusao
            prediction = []
            prediction.append(actual_class)
            prediction.append(predicted_class)
            repetition_predictions.append(prediction)

            #print("actual:", actual_class, " prediction:", predicted_class)

            # para calculo de taxa de erro
            if (actual_class == predicted_class):
                true_positives += 1
            else:
                false_positives += 1

        #print("true positives: %s" % true_positives)
        #print("false positives: %s" % false_positives)

        error_rate = util.errorRate(true_positives, false_positives)

        error_rates.append(error_rate)
        predictions.append(repetition_predictions)

    return predictions, error_rates