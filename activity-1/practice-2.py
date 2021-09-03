import numpy as np
from sklearn.datasets import load_linnerud
from sklearn.multioutput import MultiOutputRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split

fname_X = "features.pos"
fname_y = "outcomes.pof"
X = np.loadtxt(fname_X)
y = np.loadtxt(fname_y)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
clf = MultiOutputRegressor(Ridge(random_state=123)).fit(X_train, y_train)
print(clf.predict(X_test))
