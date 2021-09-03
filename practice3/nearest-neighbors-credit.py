import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors


# Function to normalize data
def normalize(atrib):
    vmax = max(atrib)
    vmin = min(atrib)
    return (atrib - vmin)/(vmax - vmin)


def nearest_neighbors(n_neighbors):
    return 0


filepath = 'credit.txt'
n_neighbors = [1, 2, 3, 5, 10, 15, 20, 50, 75, 100]

# Load credit dataset
data = pd.read_csv(filepath, sep='\t', header=0)
# student, balance and income are the inputs
student = data['student']
balance = data['balance']
income = data['income']
# Turn Yes/No values into bool values
print(student)
# # default is the output
# default = data['default']
# # Normalize inputs
# balance = normalize(balance)
# income = normalize(income)
# # Join all inputs
# X = np.stack((student, balance, income), axis=1)
