import numpy as np
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn import svm
from sklearn.svm import SVC

df = pd.read_csv('HeartAttack.csv')

x = df.drop(columns='num')
y = df['num']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=0)

# checked different classifier models like logistic regression, random forest
# but found SVM has better accuracy than others

# Load the Support Vector Machine Classifier
classifier = SVC(kernel='linear', C=1.0)   # try with different kernels
classifier.fit(x_train, y_train)
accuracy = classifier.score(x_test, y_test)

print(accuracy)

# Creating a pickle file for the classifier
filename = 'heart-attack-svm-model.pkl'
pickle.dump(classifier, open(filename, 'wb'))