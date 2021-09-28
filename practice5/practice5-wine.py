import numpy as np
import os
import matplotlib as mpl
import matplotlib.pyplot as plt
from graphviz import Source
from sklearn.tree import export_graphviz
from sklearn.datasets import load_wine
from matplotlib.colors import ListedColormap
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
import subprocess

mpl.rc('axes', labelsize=14)
mpl.rc('xtick', labelsize=12)
mpl.rc('ytick', labelsize=12)

# Figures settings
PROJECT_ROOT_DIR = "."
CHAPTER_ID = "decision_trees"
IMAGES_PATH = os.path.join(PROJECT_ROOT_DIR, "images", CHAPTER_ID)
os.makedirs(IMAGES_PATH, exist_ok=True)

# Writting console output into a file.
fnres = 'consoleResultsWine.txt'
fres = open(fnres, 'w')


# Function to create the png of the plots
def save_fig(fig_id, tight_layout=True, fig_extension="png", resolution=300):
    path = os.path.join(IMAGES_PATH, fig_id + "." + fig_extension)
    print("Saving figure", fig_id)
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format=fig_extension, dpi=resolution)


# Load the wine dataset
wine = load_wine()
X = wine.data
y = wine.target
x_train, x_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, train_size=0.8, random_state=1, shuffle=True)
tree_clf = DecisionTreeClassifier(random_state=0)

# Train the model
tree_clf.fit(x_train, y_train)

# Create the graphviz dot file for the wine dataset
export_graphviz(tree_clf, out_file=os.path.join(IMAGES_PATH, "wine_tree.dot"),
                feature_names=wine.feature_names, class_names=wine.target_names, rounded=True, filled=True)
Source.from_file(os.path.join(IMAGES_PATH, "wine_tree.dot"))

# Create the tree png
treepng = "wine_tree.png"
cmd = ["dot", "-Tpng", os.path.join(IMAGES_PATH, "wine_tree.dot"),
       "-o", os.path.join(IMAGES_PATH, treepng)]
p = subprocess.Popen(cmd)

# Print results and write them to external file
print("==============TESTING IRIS DATASET===============\n")
fres.write("==============TESTING IRIS DATASET===============\n")
testset_score = tree_clf.score(x_test, y_test)
print("The score obtained by the test sets is: {scr}".format(
    scr=testset_score))
fres.write("The score obtained by the test sets is: {scr}\n".format(
    scr=testset_score))
trainset_score = tree_clf.score(x_train, y_train)
print("The score obtained by the train sets is: {scr}".format(
    scr=trainset_score))
fres.write("The score obtained by the train sets is: {scr}\n".format(
    scr=trainset_score))

fres.close()
