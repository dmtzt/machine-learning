from sklearn import svm
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression

def printsep():
    print('\n\n------------------------------------------------------------------------------------\n\n\n')

digits = load_digits()
X = digits.data
y = digits.target
x_train, x_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, train_size=0.8, random_state=1, shuffle=True)

# SVM models
# Linear kernel: linearly separable data (faster)
linear_svc = svm.SVC(kernel='linear')
# Polynomial kernel: polynomial decision boundary using combinations of data points (less efficient)
poly_svc = svm.SVC(kernel='poly')
# Radial Basis Function (RBF) kernel: non linearly separable data (most popular and efficient)
rbf_svc = svm.SVC(kernel='rbf')
# Sigmoid kernel: uses sigmoid function, widely used in neural networks
sigmoid_svc = svm.SVC(kernel='sigmoid')

# Train SVM models
linear_svc.fit(x_train, y_train)
poly_svc.fit(x_train, y_train)
rbf_svc.fit(x_train, y_train)
sigmoid_svc.fit(x_train, y_train)

# Get SVM models predictions
linear_pred = linear_svc.predict(x_test)
poly_pred = poly_svc.predict(x_test)
rbf_pred = rbf_svc.predict(x_test)
sigmoid_pred = sigmoid_svc.predict(x_test)

# Get SVM models precision scores
linear_ps = linear_svc.score(x_test, y_test)
poly_ps = poly_svc.score(x_test, y_test)
rbf_ps = rbf_svc.score(x_test, y_test)
sigmoid_ps = sigmoid_svc.score(x_test, y_test)

# Get SVM models confusion matrices
linear_cmat = confusion_matrix(linear_pred, y_test)
poly_cmat = confusion_matrix(poly_pred, y_test)
rbf_cmat = confusion_matrix(rbf_pred, y_test)
sigmoid_cmat = confusion_matrix(sigmoid_pred, y_test)

# Logistic regression model
# Usar valor 'ovr'
log_model = LogisticRegression(multi_class='ovr')
log_model.fit(x_train,y_train)
log_pred = log_model.predict(x_test)
log_ps = log_model.score(x_test,y_test)
log_model_cmat = confusion_matrix(log_pred, y_test)

# k-NN models
# Neighbors, K, precision score and prediction with best performance
n_neighbors_values = [1, 2, 3, 5, 10, 15, 20, 50, 75, 100]
k_best = 0
ps_best = 0
pred_best = []
cmat_best = []
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
    ps = nbrs.score(x_test, y_test)
    # Compute confusion matrix to evaluate accuracy
    cmat = confusion_matrix(pred, y_test)

    # Find k with best precision score
    if (ps > ps_best):
        k_best = k
        ps_best = ps
        pred_best = pred
        cmat_best = cmat
        knn_bestmodel = nbrs

# Naive-Bayes model
nb_model = GaussianNB()
nb_model.fit(x_train,y_train)
nbpred = nb_model.predict(x_test)
nb_ps = nb_model.score(x_test,y_test)
nb_model_cmat = confusion_matrix(nbpred, y_test)

# Print scores and confusion matrices
printsep()
print(
    'Linear SVM - ps: {ps} - confusion matrix:\n{cmat}'.format(ps=linear_ps, cmat=linear_cmat))
printsep()
print(
    'Polynomial SVM - ps: {ps} - confusion matrix:\n{cmat}'.format(ps=poly_ps, cmat=poly_cmat))
printsep()
print(
    'RBF SVM - ps: {ps} - confusion matrix:\n{cmat}'.format(ps=rbf_ps, cmat=rbf_cmat))
printsep()
print(
    'Sigmoid SVM - ps: {ps} - confusion matrix:\n{cmat}'.format(ps=sigmoid_ps, cmat=sigmoid_cmat))
printsep()
print(
    'Logistic Regression - ps: {ps} - confusion matrix:\n{cmat}'.format(ps=log_ps, cmat=log_model_cmat))
printsep()
print(
    'k-NN - best k: {k} - ps: {ps} - confusion matrix:\n{cmat}'.format(k=k_best, ps=ps_best, cmat=cmat_best))
printsep()
print(
    'Naive Bayes - ps: {ps} - confusion matrix:\n{cmat}'.format(ps=nb_ps, cmat=nb_model_cmat))



