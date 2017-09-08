import numpy as np

'''
	READS DATASETS AND CONVERTS TO A NUMPY ARRAY
'''
def readDataset(dataset_name):
	aux = []
	with open(dataset_name) as file:
	    aux = [[float(value) for value in line.split()] for line in file]

	aux = np.array(aux)

	return aux
