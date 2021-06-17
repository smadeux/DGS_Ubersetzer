import pandas as pd
import numpy as np
from sklearn import model_selection
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import validation_curve
from sklearn.datasets import load_iris
from sklearn.linear_model import Ridge


import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve
from sklearn.svm import SVC

def confusion_matrix_test(model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)
    predicted = model.predict(X_test)
    matrix = confusion_matrix(y_test, predicted)
    num_of_letters = 0
    total_percent = 0
    for i, letter in enumerate(matrix):
        num_of_letters += 1
        sum = np.sum(letter)
        percent = (letter[i]/sum)*100
        total_percent += percent
        print('{}: {:.4f}% Correct'.format(i, percent))
    avg_percent = total_percent/num_of_letters
    print('Avg Percent Correct: {:.4f}'.format(avg_percent))
    print(matrix)

def accuracy_score_test(model, X_test, y_test):
    yhat = model.predict(X_test)
    print(accuracy_score(y_test, yhat))

def split_and_create_model(dataframe):
    array = dataframe.to_numpy()
    X = array[:,1:]
    y = array[:,0]
    X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.3, random_state=1234)
    model = LogisticRegression(solver='liblinear')

    return X_train, X_test, y_train, y_test, model

# kfold = model_selection.KFold(n_splits=5, random_state=7, shuffle=True)

# np.random.seed(0)
# X, y = load_iris(return_X_y=True)
# indices = np.arange(y.shape[0])
# np.random.shuffle(indices)
# X, y = X[indices], y[indices]

# train_scores, valid_scores = validation_curve(Ridge(), X, y, "alpha",
#                                               np.logspace(-7, 3, 3),
#                                               cv=5)

# train_sizes, train_scores, valid_scores = learning_curve(SVC(kernel='linear'), X, y, train_sizes=[50, 80, 110], cv=5)

