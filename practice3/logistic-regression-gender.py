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



#Load the gender dataset
# height and weight are inputs
#gender is the output
fname = 'gender.txt'
fnres = 'Results logreg gender.txt'
gender = np.loadtxt(fname, usecols=0, dtype='S8', delimiter=',', skiprows=1)
height = np.loadtxt(fname, usecols=1, dtype=float, delimiter=',', skiprows=1)
weight = np.loadtxt(fname, usecols=2, dtype=float, delimiter=',', skiprows=1)
fres = open(fnres, 'w')
print(gender) #Male = False; Female = True
print(height)
print(weight)

#turning string values into bool values
gender = (gender == b'"Female"')
print(gender)

#Normalize data
height = normalize(height)
weight = normalize(weight)

#Joining all inputs
X = np.stack((height,weight), axis=1)


#Making the 80%-20% sets
x_train_d2, x_test_d2,y_train_d2, y_test_d2 =train_test_split(X,gender, test_size=0.2, train_size=0.8, random_state=1, shuffle=True)
#fit the linear model
model_d2 = LogisticRegression().fit(x_train_d2,y_train_d2)

#Get prediction
pred_d2 = model_d2.predict(x_test_d2)

#precision score:
ps = precision_score(pred_d2,y_test_d2)
print('\n*********************************\nPrecision got form sklearn.logisticReg: {ps}'.format(ps=ps))
fres.write('\n*********************************\nPrecision got form sklearn.logisticReg: {ps}\n'.format(ps=ps))
#Confusion Matrix:
cmat = confusion_matrix(pred_d2,y_test_d2)
print('Confusion Matirx:')
print(cmat)
fres.write('\n*******************************\nConfusion Matirx:\n')
fres.write(np.array2string(cmat))

for i in range(len(y_test_d2)):
    if y_test_d2[i]:
        gen = 'ro'
    else:
        gen = 'bo'
    plt.plot(x_test_d2[i][0],x_test_d2[i][1],gen)
plt.xlabel('Heights')
plt.ylabel('Weights')
plt.title('Actual Height vs Weight')
#plt.show()
plt.savefig('plot_actual_gender_weight_height_logreg.png', dpi=300, bbox_inches='tight')
plt.clf()
for i in range(len(pred_d2)):
    if pred_d2[i]:
        gen = 'ro'
    else:
        gen = 'bo'
    plt.plot(x_test_d2[i][0],x_test_d2[i][1],gen)
plt.xlabel('Heights')
plt.ylabel('Weights')
plt.title('Predicted Height vs Weight')
#plt.show()
plt.savefig('plot_predicted_gender_weight_height_logreg.png', dpi=300, bbox_inches='tight')

fres.close()