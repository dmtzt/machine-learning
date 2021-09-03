# Practice 1, pt 2
# Linear regression with n-fold validation
# Machine Learning group 500
# 23/08/2021
# Luis Fernando Lomelin Ibarra
# David Alejandro Martinez Tristan

import numpy as nup
import cupy as np
from numpy.core.fromnumeric import mean
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from io import StringIO

#Definir batch gradient decent
def batch_gradient_decent_MSE(X, y, a = 0.01,  ep = 0.0001, maxIter = 10000):
    converge = False
    #beta = np.ones((X.shape[1],1)) #create a beta based on the number of features
    beta = np.ones(X.shape[1])
    n = X.shape[0]
    print('Shape X: ')
    print(X.shape)
    print('Beta shape:')
    print(beta.T.shape)
    #Nuestras formulas asumen las formas (fetures, samples)
    x_t =  X.transpose()
    it = 0
    #h(beta) = beta^T . xi -> vectorizado = h(beta) = X . beta
    J = 1/n * np.sum(np.power((np.dot(X,beta)-y),2))
    print(J)
    while not converge:
        #betaj+1 = bj + a * grad
        grad = 2/n *  np.dot(x_t,   ( np.dot(X,beta) - y))
        beta = beta - a * grad
        print('beta = {bt}'.format(bt=beta))
        #Getting the error
        err =1/n * np.sum(np.power((np.dot(X,beta)-y),2))
        print('Err = {er}'.format(er=err))
        #Check if converges
        if abs(J - err) <= ep:
            print('Converged')
            converge =True
        #Check if exceeded max iterations
        if it >= maxIter:
            print('Max iterations exceeded')
            converge = True
        it = it +1
        J = err
    return beta

# Load dataset
fname = 'clean_gendata.txt'
print("Filename: {fname}".format(fname=fname))
# Input: Height
X = nup.loadtxt(fname, skiprows=1, usecols=(0)).reshape((-1,1))
# Output: Weight
y = nup.loadtxt(fname, skiprows=1, usecols=(1))

#Create the test and training arrays using 80/20 split
x_train, x_test,y_train, y_test=train_test_split(X,y, test_size=0.2, train_size=0.8, random_state=1, shuffle=True)
#fit the linear model
reg = LinearRegression().fit(x_train,y_train)
#Get prediction
predictions = reg.predict(x_test)
#get MSE
mse = mean_squared_error(y_test, predictions)

print('The MSE of the model was: {mse}'.format(mse=mse))

#using batch gradient decent

#Defining the learning rate
alpha = 0.0001
x_train = np.array(x_train)
y_train = np.array(y_train)
beta = batch_gradient_decent_MSE(x_train, y_train ,alpha)
nup.savetxt("beta.txt",np.asnumpy(beta))
n = X.shape[0]
x_test = np.array(x_test)
y_test = np.array(y_test)
X = x_test.transpose()
y = y_test.transpose()
J = 1/n * np.sum(np.power((np.dot(beta,X)-y),2))
print('MSE using gradient decent: {mse}'.format(mse=J))




