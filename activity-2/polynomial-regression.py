import pandas as pd
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LinearRegression
from sklearn.metrics import precision_score
from sklearn.model_selection import train_test_split

degree = 9
fname = 'polynomial.txt'

# x: age, y: length
data = pd.read_csv(fname, sep='\t', header=0)
age = data['age']
length = data['length']

# fit the polynomial model
polyreg = make_pipeline(PolynomialFeatures(degree), LinearRegression())
polyreg.fit(age_train, length_train)
age_pred = polyreg.predict(age_test)

# precision score
ps = precision_score(age_pred, age_test)
print(ps)
