import util
import numpy as np
from sklearn.neighbors.kde import KernelDensity
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold

'''
	CLASSIFICADOR I
	JANELA DE PARZEN
	KERNEL MULTIVARIADA DO PRODUTO
'''

def gaussian_kernel(h, d, x, x_i):
    # utilizar a funcao de kernel multivariada do produto
    # ao inves de ssa gaussian kernel
    return (1 / (((2 * np.pi)**d) * (h**d))) * np.exp(-1/2 * (((x - x_i) / h)**2))

def parzen_window_func(phi, h_d, n):
    return 1/(n * h_d) * np.sum(phi)

def parzen_estimation(x_samples, point, h, d, kernel_func, window_func):
    n = x_samples.shape[0]
    v = h**d
    k = 0
    for sample in x_samples:
        phi = kernel_func(h, d, sample, point)
        k += window_func(phi, v, n)

    return (k / n) / v

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

skf = StratifiedKFold(n_splits=2, random_state=42)

for train_index, test_index in skf.split(fac, target):

    train_set = fac[train_index]
    train_target = target[train_index]
    test_set = fac[test_index]
    test_target = target[test_index]

    # training set and class sample size
    test_sample_size = test_set.shape[0]
    class_sample_size = test_sample_size / 10

    # iterates through class samples
    for w in range(0, 10):
        # limits for training samples from class
        initial_sample = w * int(class_sample_size)
        end_sample = initial_sample + int(class_sample_size)

        samples = train_set[initial_sample:end_sample, :]
        test_samples = train_target[initial_sample:end_sample]

        point_x = samples[1,:]

        print('class', w, 'p(x) =', parzen_estimation(samples, point_x, h=1, d=samples.shape[1],
                                          kernel_func=gaussian_kernel,
                                          window_func=parzen_window_func
                                          ))

    print("======================= FINISH CV =============================")