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


# Turn 'Female'/'Male' strings into boolean values
def female_male_boolean(array):
    return (array == 'Female')


filepath = 'gender.txt'
n_neighbors_values = [1, 2, 3, 5, 10, 15, 20, 50, 75, 100]
fnres = 'full results gender.txt'
fres = open(fnres, 'w')
fres.write('*******************************************************\n**** K-NN\n**************************************************************************\n')
# Load gender dataset
data = pd.read_csv(filepath, header=0)
# height and weight are the inputs
height = data['Height']
weight = data['Weight']
# gener is the output
gender = data['Gender']

# Turn 'Female'/'Male' strings into boolean values
gender = female_male_boolean(gender)

# Normalize inputs
height = normalize(height)
weight = normalize(weight)

# Join all inputs
X = np.stack((height, weight), axis=1)

# Make train and test sets (80%-20%)
x_train, x_test, y_train, y_test = train_test_split(
    X, gender, test_size=0.2, train_size=0.8, random_state=1, shuffle=True)

# K and ps values lists
k_list = []
ps_list = []

# K, precision score and prediction with best performance
k_best = 0
ps_best = 0
pred_best = []
knn_bestmodel = None

# Plot actual gender - height vs weight instances from test set
plt.title('Actual gender - Height vs Weight')
plt.xlabel('height')
plt.ylabel('weight')

for idx, y in enumerate(y_test):
    # print('idx: {idx}, y: {y}'.format(idx=idx, y=y))
    # Red if instance is female, blue if it is male
    gender = 'ro' if (y) else 'bo'
    plt.plot(x_test[idx][0], x_test[idx][1], gender)

# Save plot
plt.savefig('plot_actual_gender_weight_height.png',
            dpi=300, bbox_inches='tight')
plt.clf()

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
        pred_best = pred
        knn_bestmodel = nbrs

    # Append k value and precision score to their lists
    k_list.append(k)
    ps_list.append(ps)

print('best k: {k_best}, ps: {ps_best}'.format(k_best=k_best, ps_best=ps_best))
fres.write('best k: {k_best}, ps: {ps_best}\n'.format(k_best=k_best, ps_best=ps_best))

# Plot height vs weight instances from prediction with best precision
plt.title('Predicted gender - Height vs Weight')
plt.xlabel('height')
plt.ylabel('weight')

for idx, y in enumerate(pred_best):
    gender = 'ro' if (y) else 'bo'
    plt.plot(x_test[idx][0], x_test[idx][1], gender)

# Save plot
plt.savefig('plot_predicted_gender_weight_height.png',
            dpi=300, bbox_inches='tight')
plt.clf()

# Plot k-neighbors vs precision rate scatterplot
plt.title('k-neighbors vs precision rate')
plt.xlabel('k-neighbors')
plt.ylabel('precision score')

plt.scatter(k_list, ps_list)
plt.plot(k_list, ps_list)

# Save plot
plt.savefig('plot_gender_k-neighbors_precision_score.png',
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
plt.savefig('plot_gender_k-nn-ROC.png',
            dpi=300, bbox_inches='tight')
plt.clf()

#LOGREG
fres.write('*******************************************************\n**** Logistic Regresion\n**************************************************************************\n')
#fit the Logistic Regresion model
model_logreg = LogisticRegression().fit(x_train,y_train)

#Get prediction
pred_logreg = model_logreg.predict(x_test)

#precision score:
#ps = precision_score(pred_logreg,y_test)
ps = model_logreg.score(x_test, y_test)
print('\n*********************************\nPrecision got form sklearn.logisticReg: {ps}'.format(ps=ps))
fres.write('\n*********************************\nPrecision got form sklearn.logisticReg: {ps}\n'.format(ps=ps))
#Confusion Matrix:
cmat = confusion_matrix(pred_logreg,y_test)
print('Confusion Matirx:')
print(cmat)
fres.write('\n*******************************\nConfusion Matirx:\n')
fres.write(np.array2string(cmat))


for i in range(len(pred_logreg)):
    if pred_logreg[i]:
        gen = 'ro'
    else:
        gen = 'bo'
    plt.plot(x_test[i][0],x_test[i][1],gen)
plt.xlabel('Heights')
plt.ylabel('Weights')
plt.title('Predicted Height vs Weight')
#plt.show()
plt.savefig('plot_predicted_gender_weight_height_logreg.png', dpi=300, bbox_inches='tight')

#ROC curve for logreg

knn_pred_p_best = model_logreg.predict_proba(x_test)[:,1]
fpr, tpr, thresholds = roc_curve(y_test, knn_pred_p_best)
print('ROC analisis: True Positive Rate = {tpr} ; False Positive Rate: {fpr}; Threadhold = {th}'.format(tpr=tpr,fpr=fpr,th=thresholds))
fres.write('ROC analisis: True Positive Rate = {tpr} ; False Positive Rate: {fpr}; Threadhold = {th}\n'.format(tpr=tpr,fpr=fpr,th=thresholds))
plot_roc_curve(model_logreg, x_test, y_test)
plt.title("ROC curve LogReg")
plt.savefig('plot_gender_LogReg-ROC.png',
            dpi=300, bbox_inches='tight')
plt.clf()
fres.close()
