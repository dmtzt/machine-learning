from sklearn import svm
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

digits = load_digits()
X = digits.data
y = digits.target
x_train, x_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, train_size=0.8, random_state=1, shuffle=True)

linear_svc = svm.SVC(kernel='linear')
poly_svc = svm.SVC(kernel='poly')
rbf_svc = svm.SVC(kernel='rbf')
sigmoid_svc = svm.SVC(kernel='sigmoid')

# Train models
linear_svc.fit(X, y)
poly_svc.fit(X, y)
rbf_svc.fit(X, y)
sigmoid_svc.fit(X, y)
