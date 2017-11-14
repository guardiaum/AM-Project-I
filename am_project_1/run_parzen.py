import util
import nayve_bayes as bayes
import numpy as np
from sklearn.neighbors.kde import KernelDensity
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold

'''
	CLASSIFICADOR I
	JANELA DE PARZEN
	FUNCAO KERNEL MULTIVARIADA DO PRODUTO
'''

# multivariate kernel function
# h = bandwidth
# x = point for dennsity estimation
# x_i = samples
def gaussian_kernel(h, x, x_i):
    return (x - x_i) / h


# product from kernel function
def parzen_window_func(phi):
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
        k += parzen_window_func(phi)

    density = (1/n) * (1/float(v)) * float(k)

    print("k", k)
    print("n", n)
    print("v", v)
    print("density", density)

    return density


# fits a better estimation for parameter h
def bandwidth_estimator(data):
    # use grid search cross-validation to optimize the bandwidth
    params = {'bandwidth': np.logspace(0.2, 1, 20)}
    grid = GridSearchCV(KernelDensity(), params, cv=5)
    grid.fit(data)
    return grid.best_estimator_.bandwidth

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

error_rates = []
predictions = []

r = 0
for train_index, test_index in skf.split(fac, target):

    r += 1
    print("repetition %s" % r)

    train_set = fac[train_index]
    train_target = target[train_index]
    test_set = fac[test_index]
    test_target = target[test_index]

    #h = bandwidth_estimator(train_set)
    h = 2
    print("best bandwidth: {0}".format(h))

    # training set and class sample size
    train_sample_size = train_set.shape[0]
    train_class_size = train_sample_size / 10

    true_positives = 0
    false_positives = 0
    repetition_predictions = []

    # calculates density for each test sample
    for i, test_sample in enumerate(test_set):

        # iterates through classes calculating the density
        # for x given the class w
        densities = []
        prior_ = []
        for w in range(0, numberOfClasses):

            # limits for training samples from class
            train_initial_sample = w * int(train_class_size)
            train_end_sample = train_initial_sample + int(train_class_size)
            train_samples = train_set[train_initial_sample:train_end_sample, :]

            # calculate prior for class
            prior = bayes.calculatePrior(train_class_size, train_sample_size)

            # calculates density through parzen window estimation with gaussian kernel
            density = parzen_estimation(train_samples, test_sample, h=h, d=train_samples.shape[1],
                              kernel_func=gaussian_kernel,
                              window_func=parzen_window_func)

            prior_.append(prior)
            densities.append(density)

        densities = np.array(densities)

        # calculate the evidence
        evidence = bayes.evidence(densities, prior_)

        # calculates posteriors
        # w given x
        posteriors = []
        for w in range(0, numberOfClasses):
            print("prior", prior_[w],"density", densities[w],"evidence", evidence)
            posterior = bayes.posterior(prior_[w], densities[w], evidence)
            posteriors.append(posterior)
            print("w", w, "posterior", posterior)

        posteriors = np.array(posteriors)

        # afeta exemplo a classe de maior posteriori
        max = np.argmax(posteriors)

        # usado para gerar matriz de confusao
        prediction = []
        prediction.append(test_target[i])
        prediction.append(max)
        repetition_predictions.append(prediction)

        print("actual:", test_target[i], " prediction:", max)

        # para calculo de taxa de erro
        if (test_target[i] == max):
            true_positives += 1
        else:
            false_positives += 1

    print("true positives: %s" % true_positives)
    print("false positives: %s" % false_positives)

    error_rate = util.errorRate(true_positives, false_positives)

    error_rates.append(error_rate)
    predictions.append(repetition_predictions)

print("confusion matrix")
print(util.confusionMatrix(predictions))
print("error rate average %s" % util.errorRateAverage(error_rates))