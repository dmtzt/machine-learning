import numpy as np
import os
import matplotlib as mpl
import matplotlib.pyplot as plt
from graphviz import Source
from sklearn.tree import export_graphviz
from sklearn import datasets
from matplotlib.colors import ListedColormap
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
import subprocess,shlex

mpl.rc('axes', labelsize=14)
mpl.rc('xtick', labelsize=12)
mpl.rc('ytick', labelsize=12)

PROJECT_ROOT_DIR= "."
CHAPTER_ID = "decision_trees"
IMAGES_PATH = os.path.join(PROJECT_ROOT_DIR,"images",CHAPTER_ID)
os.makedirs(IMAGES_PATH, exist_ok=True)

#Writting console output into a file.
fnres = 'consoleResults.txt'
fres = open(fnres, 'w')

#Function to create the png of the plots
def save_fig(fig_id, tight_layout=True, fig_extension="png", resolution=300):
    path = os.path.join(IMAGES_PATH,fig_id + "." + fig_extension)
    print("Saving figure", fig_id)
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format=fig_extension, dpi=resolution)

#Function to make the plot of the decision bundary
def plot_decision_boundary(clf, X, y, axes=[0, 7.5, 0, 3], iris=True, legend=False, 
plot_training=True):
    x1s = np.linspace(axes[0], axes[1], 100)
    x2s = np.linspace(axes[2], axes[3], 100)
    x1, x2 = np.meshgrid(x1s, x2s)
    X_new = np.c_[x1.ravel(), x2.ravel()]
    y_pred = clf.predict(X_new).reshape(x1.shape)
    custom_cmap = ListedColormap(['#fafab0','#9898ff','#a0faa0'])
    plt.contourf(x1, x2, y_pred, alpha=0.3, cmap=custom_cmap)
    if not iris:
        custom_cmap2 = ListedColormap(['#7d7d58','#4c4c7f','#507d50'])
        plt.contour(x1, x2, y_pred, cmap=custom_cmap2, alpha=0.8)
    if plot_training:
        plt.plot(X[:, 0][y==0], X[:, 1][y==0], "yo", label="Iris setosa")
        plt.plot(X[:, 0][y==1], X[:, 1][y==1], "bs", label="Iris versicolor")
        plt.plot(X[:, 0][y==2], X[:, 1][y==2], "g^", label="Iris virginica")
        plt.axis(axes)
    if iris:
        plt.xlabel("Petal length", fontsize=14)
        plt.ylabel("Petal width", fontsize=14)
    else:
        plt.xlabel(r"$x_1$", fontsize=18)
        plt.ylabel(r"$x_2$", fontsize=18, rotation=0)
    if legend:
        plt.legend(loc="lower right", fontsize=14)

#Loading the iris dataset
iris = datasets.load_iris()
X = iris.data[:,:2]
y = iris.target
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, train_size=0.8, random_state=1, shuffle=True)
tree_clf = DecisionTreeClassifier(random_state=0)

#Entrenar al modelo
tree_clf.fit(x_train,y_train)


#Creating the graphviz dot file for the iris dataset
export_graphviz(tree_clf,out_file=os.path.join(IMAGES_PATH,"iris_tree.dot"), feature_names=iris.feature_names[2:],class_names=iris.target_names,rounded=True,filled=True)
Source.from_file(os.path.join(IMAGES_PATH,"iris_tree.dot"))

#Creating the tree png
treepng = "iris_tree.png"
cmd = ["dot", "-Tpng" , os.path.join(IMAGES_PATH,"iris_tree.dot"), "-o" , os.path.join(IMAGES_PATH,treepng)]
#print(cmd)
p = subprocess.Popen(cmd)

print("==============TESTING IRIS DATASET===============")
fres.write("==============TESTING IRIS DATASET===============")
testset_score = tree_clf.score(x_test,y_test)
print("The score obtained by the test sets is: {scr}".format(scr=testset_score))
fres.write("The score obtained by the test sets is: {scr}".format(scr=testset_score))
trainset_score = tree_clf.score(x_train,y_train)
print("The score obtained by the train sets is: {scr}".format(scr=trainset_score)) 
fres.write("The score obtained by the train sets is: {scr}".format(scr=trainset_score)) 




#plotting the irises
plt.figure(figsize=(8, 4))
plot_decision_boundary(tree_clf, X, y)
plt.plot([2.45, 2.45], [0, 3], "k-", linewidth=2)
plt.plot([2.45, 7.5], [1.75, 1.75], "k--", linewidth=2)
plt.text(1.40, 1.0, "Depth=0", fontsize=15)
plt.text(3.2, 1.80, "Depth=1", fontsize=13)
save_fig("decision_tree_decision_boundaries_plot")
plt.show()


fres.close()