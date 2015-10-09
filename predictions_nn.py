# Attempt to train a basic neural network on this data

import pandas as pd
import matplotlib.pyplot as plt
import xgboost as xgb
import numpy as np
import os
import pdb
from pybrain.datasets import ClassificationDataSet
from pybrain.utilities import percentError
from pybrain.tools.shortcuts import buildNetwork
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.structure.modules import SoftmaxLayer


# Load the data
os.chdir('/home/osboxes/projects/titanic/input')
train_df = shuffle(pd.read_csv('train_prepped.csv', header=0),random_state=42)
test_df = shuffle(pd.read_csv('test_prepped.csv', header=0),random_state=42)

train_df = train_df.drop(["Surname", "Name", "Ticket", "Cabin"], axis=1)
test_df = test_df.drop(["Surname", "Name", "Ticket", "Cabin"], axis=1)

# perform feature selection
collist = train_df.columns.tolist()
collist.remove("Survived")
selector = SelectKBest(f_classif, k=5)
selector.fit(train_df[collist], train_df["Survived"])
# Get the raw p-values for each feature, and transform from p-values into scores
scores = -np.log10(selector.pvalues_)

# review predictive power of predictors
plt.bar(range(len(collist)), scores)
plt.xticks(range(len(collist)), collist, rotation='vertical')
#plt.show()


predictors = ["Pclass", "Sex", "Embarked", "Title", "Irish", "FamilyWomen", "Fare", "Parch", "Age", "WomanWChild",
              "FamilyCost", "Deck"]



# modelling: cross validation definition
X_train, X_test, y_train, y_test = cross_validation.train_test_split(train_df[predictors], train_df["Survived"], test_size=0.3, random_state=42)

# modelling: algorithms, parameters to use in our ensemble



# Kaggle needs the submission to have a certain format;
# see https://www.kaggle.com/c/titanic-gettingStarted/download/gendermodel.csv
# for an example of what it's supposed to look like.
submission = pd.DataFrame({ 'PassengerId': test_df['PassengerId'],
                           'Survived': ensemble_results_valid["prediction"].astype(int)})
submission.to_csv("submission.csv", index=False)
ensemble_results_valid.to_csv("ensemble_results.csv")