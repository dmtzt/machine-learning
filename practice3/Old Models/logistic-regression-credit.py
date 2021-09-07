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



# Load the gender dataset
# student, balance and income are inputs
# default is the output
fname = 'credit.txt'
fnres = 'Results logreg credit.txt'
default = np.loadtxt(fname, usecols=1, dtype='S5',  skiprows=1)
student = np.loadtxt(fname, usecols=2, dtype='S5',  skiprows=1)
balance = np.loadtxt(fname, usecols=3, dtype=float,  skiprows=1)
income = np.loadtxt(fname, usecols=4, dtype=float,  skiprows=1)
fres = open(fnres, 'w')
#checking the read data
print(default)  # Male = False; Female = True
print(student)
print(balance)
print(income)

# turning string values into bool values
default = (default == b'"Yes"')
print(default)
student = (student == b'"Yes"')
print(student)
# Normalize data
balance = normalize(balance)
income = normalize(income)

# Joining all inputs
X = np.stack((student, balance, income), axis=1)


# Making the 80%-20% sets
x_train_d2, x_test_d2, y_train_d2, y_test_d2 = train_test_split(
    X, default, test_size=0.2, train_size=0.8, random_state=1, shuffle=True)
# fit the linear model
model_d2 = LogisticRegression().fit(x_train_d2, y_train_d2)

# Get prediction
pred_d2 = model_d2.predict(x_test_d2)

# precision score:
ps = model_d2.score(x_test_d2, y_test_d2)
print('Precision got form sklearn.logisticReg: {ps}'.format(ps=ps))
fres.write('\n*********************************\nPrecision got form sklearn.logisticReg: {ps}\n'.format(ps=ps))


# Confusion Matrix:
cmat = confusion_matrix(pred_d2, y_test_d2)
print('Confusion Matirx:')
print(cmat)
fres.write('\n*******************************\nConfusion Matirx:\n')
fres.write(np.array2string(cmat))

fres.close()

