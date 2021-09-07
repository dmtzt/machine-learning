import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score
from sklearn.model_selection import train_test_split


# Normalize data in range [0, 1]
def normalize(atrib):
    vmax = max(atrib)
    vmin = min(atrib)
    return (atrib - vmin)/(vmax - vmin)


# Turn 'Yes'/'No' strings into boolean values
def yes_no_boolean(array):
    return (array == 'Yes')


filepath = 'credit.txt'
n_neighbors_values = [1, 2, 3, 5, 10, 15, 20, 50, 75, 100]

# Load credit dataset
data = pd.read_csv(filepath, sep='\t', header=0)
# student, balance and income are the inputs
student = data['student']
balance = data['balance']
income = data['income']
# default is the output
default = data['default']

# Turn 'Yes'/'No' strings into boolean values
student = yes_no_boolean(student)
default = yes_no_boolean(default)

# Normalize inputs
balance = normalize(balance)
income = normalize(income)

# Join all inputs
X = np.stack((student, balance, income), axis=1)

# Make train and test sets (80%-20%)
x_train, x_test, y_train, y_test = train_test_split(
    X, default, test_size=0.2, train_size=0.8, random_state=1, shuffle=True)

# K and ps values lists
k_list = []
ps_list = []

# Create 10 k-nearest neighbors classifiers with predetermined values
for k in n_neighbors_values:
    # Create classifier with respective k value
    nbrs = KNeighborsClassifier(n_neighbors=k, algorithm='brute')
    # Fit the classifier from the train set
    nbrs.fit(x_train, y_train)
    # Predict class labels for test test
    pred = nbrs.predict(x_test)
    # Compute precision score according to true and false positive
    #ps = precision_score(pred, y_test)
    ps = nbrs.score(x_test, y_test)
    # Compute precision matrix to evaluate accuracy
    cmat = confusion_matrix(pred, y_test)
    # Print results
    print('K:{k}\tps:{ps}\tcmat{cmat}'.format(k=k, ps=ps, cmat=cmat))

    # Append k value and precision score to their lists
    k_list.append(k)
    ps_list.append(ps)

# k-neighbors vs precision rate scatterplot
plt.title('k-neighbors vs precision rate')
plt.xlabel('k-neighbors')
plt.ylabel('precision score')

# Scatter plot of k vs ps values
plt.scatter(k_list, ps_list)
plt.plot(k_list, ps_list)

# Save plot
plt.savefig('plot_credit_k-neighbors_precision_score.png',
            dpi=300, bbox_inches='tight')
plt.clf()
