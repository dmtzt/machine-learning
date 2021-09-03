import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
# Normalize data


def normalize(atrib):
    vmax = max(atrib)
    vmin = min(atrib)
    return (atrib - vmin)/(vmax - vmin)


# sigomid function
def sigmoid_func(z):
    return 1/(1 + np.exp(-z))


# Definir batch gradient decent
def batch_gradient_decent_log(X, y, a=0.01,  ep=0.0001, maxIter=10000):
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

    # p = 1 / (1 + e^(-B . X))
    p = sigmoid_func(np.dot(X, beta))
    print(p)
    J = - (1/n) * np.sum(y * np.log(p) + (1 - y)*np.log(1-p))
    print(J)
    while not converge:
        # betaj+1 = bj + a * grad

        grad = np.dot(x_t, (p-y))
        beta = beta - a * grad
        print('beta = {bt}'.format(bt=beta))
        p = sigmoid_func(np.dot(X, beta))
        # Getting the error
        err = - (1/n) * np.sum(y * np.log(p) + (1 - y)*np.log(1-p))
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


# Load the gender dataset
# height and weight are inputs
# gender is the output
fname = 'genero.txt'
gender = np.loadtxt(fname, usecols=0, dtype='S8', delimiter=',', skiprows=1)
height = np.loadtxt(fname, usecols=1, dtype=float, delimiter=',', skiprows=1)
weight = np.loadtxt(fname, usecols=2, dtype=float, delimiter=',', skiprows=1)
print(gender)  # Male = False; Female = True
print(height)
print(weight)

# turning string values into bool values
gender = (gender == b'"Female"')
print(gender)

# Normalize data
height = normalize(height)
weight = normalize(weight)

# Joining all inputs
X = np.stack((height, weight), axis=1)


# Making the 80%-20% sets
x_train_d2, x_test_d2, y_train_d2, y_test_d2 = train_test_split(
    X, gender, test_size=0.2, train_size=0.8, random_state=1, shuffle=True)
# fit the linear model
model_d2 = LogisticRegression().fit(x_train_d2, y_train_d2)

# Get prediction
pred_d2 = model_d2.predict(x_test_d2)

# precision score:
ps = precision_score(pred_d2, y_test_d2)
print('Precision got form sklearn.logisticReg: {ps}'.format(ps=ps))
# Confusion Matrix:
cmat = confusion_matrix(pred_d2, y_test_d2)
print('Confusion Matirx:')
print(cmat)

for i in range(len(y_test_d2)):
    if y_test_d2[i]:
        gen = 'ro'
    else:
        gen = 'bo'
    plt.plot(x_test_d2[i][0], x_test_d2[i][1], gen)
plt.xlabel('Heights')
plt.ylabel('Weights')
plt.title('Actual Height vs Weight')
# plt.show()
plt.savefig('plot_actual_gender_weight_height.png',
            dpi=300, bbox_inches='tight')
plt.clf()
for i in range(len(pred_d2)):
    if pred_d2[i]:
        gen = 'ro'
    else:
        gen = 'bo'
    plt.plot(x_test_d2[i][0], x_test_d2[i][1], gen)
plt.xlabel('Heights')
plt.ylabel('Weights')
plt.title('Predicted Height vs Weight')
# plt.show()
plt.savefig('plot_predicted_gender_weight_height.png',
            dpi=300, bbox_inches='tight')

alpha = 0.009
# Gradient Decent for gender (d2)
beta = batch_gradient_decent_log(x_train_d2, y_train_d2, alpha)
# np.savetxt("beta_gender.txt",beta)
print('\n--------------------------------------\nFinal Beta:')
print(beta)
# predict values
p_d2 = sigmoid_func(np.dot(x_test_d2, beta))
print('\n\n-------------------------------\npredictions:\n')
print(p_d2)
pred_gd_d2 = (p_d2 >= 0.5)

# precision score:
ps = precision_score(pred_gd_d2, y_test_d2)
print('Precision got from gradient decent: {ps}'.format(ps=ps))
# Confusion Matrix:
cmat = confusion_matrix(pred_gd_d2, y_test_d2)
print('Confusion Matirx of gradient decent:')
print(cmat)
plt.clf()
for i in range(len(pred_d2)):
    if pred_d2[i]:
        gen = 'ro'
    else:
        gen = 'bo'
    plt.plot(x_test_d2[i][0], x_test_d2[i][1], gen)
plt.xlabel('Heights')
plt.ylabel('Weights')
plt.title('Predicted with Gradient Decent Height vs Weight')
# plt.show()
plt.savefig('plot_predicted_gradient_decent_gender_weight_height.png',
            dpi=300, bbox_inches='tight')
