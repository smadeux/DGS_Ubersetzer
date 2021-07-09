import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, RidgeClassifier, SGDClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

import AI_tests

def train_model(file_name):
    # Get data, split into trainng and testing partitions, and train the model.
    dataframe = pd.read_csv(file_name)
    array = dataframe.to_numpy()
    X = array[:,1:]
    y = array[:,0]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1234)
    model = SGDClassifier(solver='liblinear')
    model = model.fit(X_train, y_train)

    # Run tests.
    print(file_name, ": ")
    AI_tests.accuracy_score_test(model, X_test, y_test)
    AI_tests.confusion_matrix_test(model, X_train, X_test, y_train, y_test)
    print('\n')

    # Export model.
    with open('sign_lang.pkl', 'wb') as f:
        model = pickle.dump(model, f)