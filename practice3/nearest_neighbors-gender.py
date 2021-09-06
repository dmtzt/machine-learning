import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score
from sklearn.model_selection import train_test_split


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

# Load credit dataset
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

    # Find k with best precision score
    if (ps > ps_best):
        k_best = k
        ps_best = ps
        pred_best = pred

    # Append k value and precision score to their lists
    k_list.append(k)
    ps_list.append(ps)

print('best k: {k_best}, ps: {ps_best}'.format(k_best=k_best, ps_best=ps_best))

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
