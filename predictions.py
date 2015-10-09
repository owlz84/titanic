# This script shows you how to make a submission using a few
# useful Python libraries.
# It gets a public leaderboard score of 0.76077.
# Maybe you can tweak it and do better...?

import pandas as pd
import re
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score
from sklearn.utils import shuffle
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier, RandomForestRegressor
from sklearn import cross_validation
from sklearn.grid_search import GridSearchCV
from sklearn.feature_selection import SelectKBest, f_classif
import numpy as np
import os
import pdb


# Load the data
os.chdir('/home/osboxes/projects/titanic/input')
train_df = shuffle(pd.read_csv('train_prepped.csv', header=0),random_state=42)
test_df = shuffle(pd.read_csv('test_prepped.csv', header=0),random_state=42)

# perform feature selection
collist = train_df.columns.tolist()
collist.remove("Survived")
selector = SelectKBest(f_classif, k=5)
selector.fit(train_df[collist], train_df["Survived"])
# Get the raw p-values for each feature, and transform from p-values into scores
scores = -np.log10(selector.pvalues_)

# review predictive power of predictors
plt.bar(range(len(predictors)), scores)
plt.xticks(range(len(predictors)), predictors, rotation='vertical')
plt.show()
predictors = ["Pclass", "Sex", "Embarked", "Title", "Irish", "FamilyWomen", "Fare", "Parch", "Age", "WomanWChild",
              "FamilyCost", "Deck"]


# modelling: cross validation definition
X_train, X_test, y_train, y_test = cross_validation.train_test_split(train_df[predictors], train_df["Survived"], test_size=0.3, random_state=42)

# modelling: algorithms, parameters to use in our ensemble
algorithms = [
    ["RFC", RandomForestClassifier(n_estimators=200, class_weight="auto")],
    ["RFR", RandomForestRegressor(n_estimators=100, oob_score=True, random_state=42)],
    ["Logit", LogisticRegression(random_state=1)],
    ["GradientBoost", GradientBoostingClassifier()],
    ["XGB", xgb.XGBClassifier(max_depth=3, n_estimators=300, learning_rate=0.05)]
]

# start collecting results from each model
ensemble_results_test = pd.DataFrame(y_test)
ensemble_results_valid = pd.DataFrame(test_df["PassengerId"])

for name, alg in algorithms:
    # Fit the algorithm on the training data.
    alg.fit(X_train, y_train)
    predictions_train = alg.predict(X_train)
    accuracy_train = np.mean((predictions_train> 0.5).astype(int) == y_train)
    predictions_test = alg.predict(X_test)
    accuracy_test = np.mean((predictions_test > 0.5).astype(int) == y_test)
    print("Algorithm: ", name, "Train_set accuracy: ", accuracy_train, "Test_set accuracy: ",accuracy_test)
    ensemble_results_test[name] = predictions_test
    predictions_valid = alg.predict(test_df[predictors])
    ensemble_results_valid[name] = predictions_valid

collist = ensemble_results_test.columns.tolist()
collist.remove("Survived")

# predictions based on mean
#ensemble_results["mean"] = ensemble_results[collist].astype(float).mean(axis=1)
#ensemble_results["prediction"] = (ensemble_results["mean"] > 0.5).astype(int)

# predictions based on majority vote
ensemble_results_test[collist] = ensemble_results_test[collist].astype(float)
ensemble_results_test[collist] = (ensemble_results_test[collist] > 0.5).astype(int)
ensemble_results_test["prediction"] = ensemble_results_test[collist].mode(axis=1)[0]

ensemble_results_valid[collist] = (ensemble_results_valid[collist] > 0.5).astype(int)
ensemble_results_valid["prediction"] = ensemble_results_valid[collist].mode(axis=1)[0]

accuracy_test = np.mean(ensemble_results_test["prediction"] == y_test)
print("Ensemble test_set accuracy: ", accuracy_test)

# Kaggle needs the submission to have a certain format;
# see https://www.kaggle.com/c/titanic-gettingStarted/download/gendermodel.csv
# for an example of what it's supposed to look like.
submission = pd.DataFrame({ 'PassengerId': test_df['PassengerId'],
                           'Survived': ensemble_results_valid["prediction"]})
submission.to_csv("submission.csv", index=False)