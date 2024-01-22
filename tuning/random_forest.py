import numpy as np;
import pandas as pd;
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV


SEED = 1212

train = pd.read_csv('./data/processed_train.csv')
test = pd.read_csv('./data/processed_test.csv')

X_train=train.iloc[:,1:]
y_train=train['Survived'].values
X_test = test

# Criteria for random forest
criterion=['gini', 'entropy', 'log_loss']
# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]
# Number of features to consider at every split
max_features = ['log2', 'sqrt', None]
# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(1, 10, num = 10)]
max_depth.append(None)
# Minimum number of samples required at each leaf node
min_samples_leaf = [4, 5, 6]
# Minimum number of samples required to split a node
min_samples_split = [5, 6, 7]

random_grid = {'criterion': criterion,
               'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf}

clf = RandomForestClassifier()
clf_random = RandomizedSearchCV(estimator = clf, param_distributions = random_grid, n_iter = 100, cv = 5, verbose=2, random_state=SEED, n_jobs = -1)
clf_random.fit(X_train, y_train)
print(clf_random.best_params_)

# {'n_estimators': 1000, 'min_samples_split': 6, 'min_samples_leaf': 4, 'max_features': None, 'max_depth': 7, 'criterion': 'log_loss'}