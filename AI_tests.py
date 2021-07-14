import pandas as pd
import numpy as np
from sklearn import model_selection
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import validation_curve
from sklearn.datasets import load_iris
from sklearn.linear_model import Ridge

from sklearn.model_selection import learning_curve

letters = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H',
           'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P',
           'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X',
           'Y', 'Z', 'junk']

def confusion_matrix_test(model, X_train, X_test, y_train, y_test):
    print("Confution Matrix Test")
    predicted = model.predict(X_test)
    matrix = confusion_matrix(y_test, predicted)
    num_of_letters = 0
    total_percent = 0
    for i, letter in enumerate(matrix):
        num_of_letters += 1
        sum = np.sum(letter)
        percent = (letter[i]/sum)*100
        total_percent += percent
        print('{}: {:.4f}% Correct'.format(letters[i], percent))
    avg_percent = total_percent/num_of_letters
    print('Avg Percent Correct: {:.4f}'.format(avg_percent))
    print(matrix)
    return "Confusion Matrix Test\nAvg Percent Correct: {:.4f}".format(avg_percent)

def accuracy_score_test(model, X_test, y_test):
    print("Accuracy Score Test")
    yhat = model.predict(X_test)
    accuracy = accuracy_score(y_test, yhat)
    print(accuracy)
    return "Accuracy Score Test: {:.4f}".format(accuracy)

def split_and_create_model(dataframe):
    array = dataframe.to_numpy()
    X = array[:,1:]
    y = array[:,0]
    X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.3, random_state=1234)
    model = LogisticRegression(solver='liblinear')

    return X_train, X_test, y_train, y_test, model

