# Practice 1, pt 2
# Linear regression with n-fold validation
# Machine Learning group 500
# 23/08/2021
# Luis Fernando Lomelin Ibarra
# David Alejandro Martinez Tristan

import numpy as np
from numpy.core.fromnumeric import mean
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from io import StringIO


def batch_gradient_decent_MSE(X, y, a=0.01,  ep=0.0001, maxIter=10000):
    converge = False
    # beta = np.ones((X.shape[1],1)) #create a beta based on the number of features
    beta = np.ones(X.shape[1])
    n = X.shape[0]
    print('Shape X: ')
    print(X.shape)
    print('Beta shape:')
    print(beta.T.shape)
    # Nuestras formulas asumen las formas (fetures, samples)
    x_t = X.transpose()
    it = 0
    # h(beta) = beta^T . xi -> vectorizado = h(beta) = X . beta
    J = 1/n * np.sum(np.power((np.dot(X, beta)-y), 2))
    print(J)
    while not converge:
        # betaj+1 = bj + a * grad
        grad = 2/n * np.dot(x_t,   (np.dot(X, beta) - y))
        beta = beta - a * grad
        print('beta = {bt}'.format(bt=beta))
        # Getting the error
        err = 1/n * np.sum(np.power((np.dot(X, beta)-y), 2))
        print('Err = {er}'.format(er=err))
        # Check if converges
        if abs(J - err) <= ep:
            print('Converged')
            converge = True
        # Check if exceeded max iterations
        if it >= maxIter:
            print('Max iterations exceeded')
            converge = True
        it = it + 1
        J = err
    return beta


# Load dataset
fname = 'mtcars.txt'
print("Filename: {fname}".format(fname=fname))
# Input: disp, wt
X = np.loadtxt(fname, skiprows=1, usecols=(3, 6))
# Output: hp
y = np.loadtxt(fname, skiprows=1, usecols=(4))

# Dataset of 32 instances, n-fold cross validation will be applied given its length
kf = KFold(n_splits=len(X), shuffle=True)
kf.get_n_splits(X)

# global precision
precision = 0

# Build n different linear models from dataset splits
for train_index, test_index in kf.split(X):
    #print("train:", train_index, "test:", test_index)
    # Get train and test datasets
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    # Fit linear model
    reg = LinearRegression().fit(X_train, y_train)
    # Get prediction
    y_pred = reg.predict(X_test)
    # Get model MSE
    mse = mean_squared_error(y_test, y_pred)
    # Append MSE to global precision
    precision = precision + mse
    # Print MSE
    print(
        'a = {a}, b = {b}, MSE:{mse}'.format(a=reg.coef_[0], b=reg.coef_[1], mse=mse))

# Calculate global precision
precision = precision / len(X)
# Print global precision
print('Global precision: {precision}'.format(precision=precision))

# Calculate batch gradient decent
alpha = 0.0001
beta = batch_gradient_decent_MSE(X_train, y_train, alpha)

n = X.shape[0]
X = X_test.transpose()
y = y_test.transpose()
J = 1/n * np.sum(np.power((np.dot(beta, X)-y), 2))
print('MSE using gradient decent: {mse}'.format(mse=J))
