import numpy as np;
import pandas as pd;
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.model_selection import RandomizedSearchCV


SEED = 1212

train = pd.read_csv('./data/processed_train.csv')
test = pd.read_csv('./data/processed_test.csv')

X_train=train.iloc[:,1:]
y_train=train['Survived'].values
X_test = test

# Regularization Parameter
C = [0.01, 0.1, 1]
# Kernel Type
kernel = ['poly', 'rbf']
# Degree of polynomial
degree = [1, 2]
# L2 regularization term
gamma = [0.1,0.01,0.001]
gamma.append('scale')
gamma.append('auto')


random_grid = {'C':C,
               'kernel': kernel,
               'degree': degree,
               'gamma': gamma}

clf = SVC()
clf_random = RandomizedSearchCV(estimator = clf, param_distributions = random_grid, n_iter = 100, cv = 5, verbose=2, random_state=SEED, n_jobs = -1)
clf_random.fit(X_train, y_train)
print(clf_random.best_params_)
# {'kernel': 'poly', 'gamma': 'auto', 'degree': 2, 'C': 1}