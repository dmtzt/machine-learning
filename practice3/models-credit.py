import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve
from sklearn.metrics import plot_roc_curve

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

fnres = 'full results credit.txt'
fres = open(fnres, 'w')
fres.write('*******************************************************\n**** K-NN\n**************************************************************************\n')

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
ps_best = 0
k_best = -10
knn_bestmodel = None

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
    fres.write('K:{k}\tps:{ps}\tcmat{cmat}\n'.format(k=k, ps=ps, cmat=cmat))

    # Find k with best precision score
    if (ps > ps_best):
        k_best = k
        ps_best = ps
        knn_bestmodel = nbrs
    # Append k value and precision score to their lists
    k_list.append(k)
    ps_list.append(ps)

print('best k: {k_best}, ps: {ps_best}'.format(k_best=k_best, ps_best=ps_best))
fres.write('best k: {k_best}, ps: {ps_best}\n'.format(k_best=k_best, ps_best=ps_best))
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

#ROC curve for k-nn

knn_pred_p_best = knn_bestmodel.predict_proba(x_test)[:,1]
#print(knn_pred_p_best)
fpr, tpr, thresholds = roc_curve(y_test, knn_pred_p_best)
print('ROC analisis: True Positive Rate = {tpr} ; False Positive Rate: {fpr}; Threadhold = {th}'.format(tpr=tpr,fpr=fpr,th=thresholds))
fres.write('ROC analisis: True Positive Rate = {tpr} ; False Positive Rate: {fpr}; Threadhold = {th}\n'.format(tpr=tpr,fpr=fpr,th=thresholds))
plot_roc_curve(knn_bestmodel, x_test, y_test)
plt.title("ROC curve k-NN")
plt.savefig('plot_credit_k-nn-ROC.png',
            dpi=300, bbox_inches='tight')
plt.clf()

# fit the logreg model
model_logreg = LogisticRegression().fit(x_train, y_train)

# Get prediction
pred_logreg = model_logreg.predict(x_test)
fres.write('*******************************************************\n**** Logistic Regresion\n**************************************************************************\n')
# precision score:
ps = model_logreg.score(x_test, y_test)
print('Precision got form LogReg: {ps}'.format(ps=ps))
fres.write('\n*********************************\nPrecision got form sklearn.logisticReg: {ps}\n'.format(ps=ps))


# Confusion Matrix for logreg:
cmat = confusion_matrix(pred_logreg, y_test)
print('Confusion Matirx:')
print(cmat)
fres.write('\n*******************************\nConfusion Matirx LogReg:\n')
fres.write(np.array2string(cmat))

#ROC curve for logreg

knn_pred_p_best = model_logreg.predict_proba(x_test)[:,1]
fpr, tpr, thresholds = roc_curve(y_test, knn_pred_p_best)
print('ROC analisis: True Positive Rate = {tpr} ; False Positive Rate: {fpr}; Threadhold = {th}'.format(tpr=tpr,fpr=fpr,th=thresholds))
fres.write('ROC analisis: True Positive Rate = {tpr} ; False Positive Rate: {fpr}; Threadhold = {th}\n'.format(tpr=tpr,fpr=fpr,th=thresholds))
plot_roc_curve(model_logreg, x_test, y_test)
plt.title("ROC curve LogReg")
plt.savefig('plot_credit_LogReg-ROC.png',
            dpi=300, bbox_inches='tight')
plt.clf()
fres.close()