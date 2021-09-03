import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score
from sklearn.model_selection import train_test_split

#Normalize data
def normalize(atrib):
    vmax = max(atrib)
    vmin = min(atrib)
    return (atrib - vmin)/(vmax - vmin)

#sigomid function
def sigmoid_func(z):
    return 1/(1 + np.exp(-z))



#Definir batch gradient decent
def batch_gradient_decent_log(X, y, a = 0.01,  ep = 0.0001, maxIter = 10000):
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
    
    # p = 1 / (1 + e^(-B . X))
    p = sigmoid_func(np.dot(X,beta))
    print(p)
    J =  - (1/n) * np.sum(y * np.log(p) + (1 - y)*np.log(1-p))
    print(J)
    while not converge:
        #betaj+1 = bj + a * grad
        
        grad = np.dot(x_t,(p-y))
        beta = beta - a * grad
        print('beta = {bt}'.format(bt=beta))
        p = sigmoid_func(np.dot(X,beta))
        #Getting the error
        err = - (1/n) *  np.sum(y * np.log(p) + (1 - y)*np.log(1-p))
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

#Load default data set
#Student, balance and income are inputs
#default are outputs
fname = 'default.txt'

default = np.loadtxt(fname, usecols=1, dtype='S5', delimiter='\t', skiprows=1)
student = np.loadtxt(fname, usecols=2, dtype='S5', delimiter='\t', skiprows=1)
balance = np.loadtxt(fname, usecols=3, dtype=float, delimiter='\t', skiprows=1)
income = np.loadtxt(fname, usecols=4, dtype=float, delimiter='\t', skiprows=1)

print(default)
print(student)
print(balance)
print(income)

#turning string values into bool values
student = (student == b'"Yes"')
print(student)
default = (default == b'"Yes"')

#Normalizing inputs
#student = normalize(student)
balance = normalize(balance)
income = normalize(income)
#Joining all inputs
X = np.stack((student, balance, income), axis=1)


#Making the 80%-20% sets
x_train_d1, x_test_d1,y_train_d1, y_test_d1 =train_test_split(X,default, test_size=0.2, train_size=0.8, random_state=1, shuffle=True)
#fit the linear model
model_d1 = LogisticRegression().fit(x_train_d1,y_train_d1)

#Get prediction
pred_d1 = model_d1.predict(x_test_d1)

#precision score:
ps = precision_score(pred_d1,y_test_d1)
print('\n*********************************\nPrecision got form sklearn.logisticReg: {ps}'.format(ps=ps))
#Confusion Matrix:
cmat = confusion_matrix(pred_d1,y_test_d1)
print('\n*******************************\nConfusion Matirx:')
print(cmat)

alpha = 0.009
#Gradient Decent for  default (d1)
beta = batch_gradient_decent_log(x_train_d1, y_train_d1 ,alpha)
np.savetxt("beta_default.txt",beta)
print('\n--------------------------------------\nFinal Beta:')
print(beta)
#predict values
p_d1 = sigmoid_func(np.dot(x_test_d1, beta))
print('\n\n-------------------------------\npredictions:\n')
print(p_d1)
pred_gd_d1 = (p_d1 >= 0.5)

#precision score:
ps = precision_score(pred_gd_d1,y_test_d1)
print('Precision got from gradient decent: {ps}'.format(ps=ps))
#Confusion Matrix:
cmat = confusion_matrix(pred_gd_d1,y_test_d1)
print('\n\nConfusion Matirx of gradient decent:')
print(cmat)


