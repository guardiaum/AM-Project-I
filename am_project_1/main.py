import readDataset as inp
import numpy as np
from scipy.spatial.distance import pdist, squareform

# path to datasets
fac_file = "mfeat/mfeat-fac"
fou_file = "mfeat/mfeat-fou"
kar_file = "mfeat/mfeat-kar"

# Get datasets as a numpy 2d array
fac = inp.readDataset(fac_file)
fou = inp.readDataset(fou_file)
kar = inp.readDataset(kar_file)

# calculates euclidean distance as a condensed distance matrix
fac_distance = pdist(fac, metric='euclidean')
fou_distance = pdist(fou, metric='euclidean')
kar_distance = pdist(kar, metric='euclidean')

# converts the vector-form distance matrix 
# to a square-form distance matrix
fac_distance_matrix = squareform(fac_distance)
fou_distance_matrix = squareform(fou_distance)
kar_distance_matrix = squareform(kar_distance)
