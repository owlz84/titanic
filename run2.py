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


def get_title(name):
    # Use a regular expression to search for a title.  Titles always consist of capital and lowercase letters, and end with a period.
    title_search = re.search(' ([A-Za-z]+)\.', name)
    # If the title exists, extract and return it.
    if title_search:
        return title_search.group(1)
    return ""


def get_deck(cabin):
    try:
        return cabin[0]
    except TypeError:
        return "unknown"


# Load the data
os.chdir('/home/osboxes/projects/titanic/input')
train_df = shuffle(pd.read_csv('train.csv', header=0),random_state=42)
test_df = shuffle(pd.read_csv('test.csv', header=0),random_state=42)
both_df = pd.concat([train_df, test_df], axis=0)
irish_names = pd.read_csv('irish_names.csv', header=0)
aristocratic_names = pd.read_csv('aristocratic_names.csv', header=0)


train_df["Title"] = train_df["Name"].apply(get_title)
test_df["Title"] = test_df["Name"].apply(get_title)
both_df["Title"] = both_df["Name"].apply(get_title)

# make the assumption that the important part is the last word in the name

title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Dr": 5, "Rev": 6,
                 "Major": 7, "Col": 7, "Mlle": 8, "Mme": 8, "Don": 9, "Lady": 10,
                 "Countess": 10, "Jonkheer": 10, "Sir": 9, "Capt": 7, "Ms": 2, "Dona": 10}

for k,v in title_mapping.items():
    train_df["Title"][train_df["Title"] == k] = v
    test_df["Title"][test_df["Title"] == k] = v
    both_df["Title"][both_df["Title"] == k] = v

train_df["Title"] = train_df["Title"].astype(int)
test_df["Title"] = test_df["Title"].astype(int)
both_df["Title"] = both_df["Title"].astype(int)

train_df["Surname"] = train_df["Name"].apply(lambda x: x.split(",")[0].lower())
test_df["Surname"] = test_df["Name"].apply(lambda x: x.split(",")[0].lower())
both_df["Surname"] = both_df["Name"].apply(lambda x: x.split(",")[0].lower())

irish_names["Surname"] = irish_names["Surname"].apply(lambda x: x.split()[-1].lower())
aristocratic_names["Surname"] = aristocratic_names["Surname"].apply(lambda x: x.split()[-1].lower())
train_df["Irish"] = np.in1d(train_df["Surname"], irish_names["Surname"]).astype(int)
train_df["Aristocrat"] = np.in1d(train_df["Surname"], aristocratic_names["Surname"]).astype(int)
test_df["Irish"] = np.in1d(test_df["Surname"], irish_names["Surname"]).astype(int)
test_df["Aristocrat"] = np.in1d(test_df["Surname"], aristocratic_names["Surname"]).astype(int)

train_df.loc[train_df["Sex"] == "male", "Sex"] = 0
train_df.loc[train_df["Sex"] == "female", "Sex"] = 1
train_df["Sex"] = train_df["Sex"].astype(int)
test_df.loc[test_df["Sex"] == "male", "Sex"] = 0
test_df.loc[test_df["Sex"] == "female", "Sex"] = 1
test_df["Sex"] = test_df["Sex"].astype(int)
both_df.loc[both_df["Sex"] == "male", "Sex"] = 0
both_df.loc[both_df["Sex"] == "female", "Sex"] = 1
both_df["Sex"] = both_df["Sex"].astype(int)

train_df["FamilyMembers"] = train_df["Surname"].apply(lambda x: both_df["Surname"][both_df["Surname"] == x].count())
test_df["FamilyMembers"] = test_df["Surname"].apply(lambda x: both_df["Surname"][both_df["Surname"] == x].count())

def get_family_women(surname):
    df = both_df[(both_df["Sex"] == 1) & (both_df["Surname"] == surname)]
    return df["Surname"].count()

def get_family_children(surname):
    df = both_df[(both_df["Age"] < 16) & (both_df["Surname"] == surname)]
    return df["Surname"].count()

train_df["FamilyWomen"] = train_df["Surname"].apply(get_family_women)
test_df["FamilyWomen"] = test_df["Surname"].apply(get_family_women)

train_df["FamilyChildren"] = train_df["Surname"].apply(get_family_children)
test_df["FamilyChildren"] = test_df["Surname"].apply(get_family_children)

train_df["WomanWChild"] = (train_df["FamilyChildren"] >= 1) & (train_df["Sex"] == 1)
test_df["WomanWChild"] = (test_df["FamilyChildren"] >= 1) & (test_df["Sex"] == 1)
train_df["WomanWChild"] = train_df["WomanWChild"].astype(int)
test_df["WomanWChild"] = test_df["WomanWChild"].astype(int)

train_df["Fare"][pd.isnull(train_df["Fare"])] = both_df["Fare"].median()
test_df["Fare"][pd.isnull(test_df["Fare"])] = both_df["Fare"].median()
both_df["Fare"][pd.isnull(both_df["Fare"])] = both_df["Fare"].median()


def get_family_fare(surname):
    return both_df["Fare"][both_df["Surname"] == surname].mean()

train_df["FamilyCost"] = train_df["Surname"].apply(get_family_fare)
test_df["FamilyCost"] = test_df["Surname"].apply(get_family_fare)


def get_median_age(title):
    if title in [2, 4]:
        return both_df["Age"][both_df["Title"] == title].median()
    else:
        return both_df["Age"].median()

train_df["Age"][pd.isnull(train_df["Age"])] = train_df["Title"].apply(get_median_age)
test_df["Age"][pd.isnull(test_df["Age"])] = test_df["Title"].apply(get_median_age)

train_df["Embarked"] = train_df["Embarked"].fillna("S")
train_df.loc[train_df["Embarked"] == "S", "Embarked"] = 0
train_df.loc[train_df["Embarked"] == "C", "Embarked"] = 1
train_df.loc[train_df["Embarked"] == "Q", "Embarked"] = 2
train_df["Embarked"] = train_df["Embarked"].astype(int)
test_df["Embarked"] = test_df["Embarked"].fillna("S")
test_df.loc[test_df["Embarked"] == "S", "Embarked"] = 0
test_df.loc[test_df["Embarked"] == "C", "Embarked"] = 1
test_df.loc[test_df["Embarked"] == "Q", "Embarked"] = 2
test_df["Embarked"] = test_df["Embarked"].astype(int)

train_df["Deck"] = train_df["Cabin"].apply(get_deck)
test_df["Deck"] = test_df["Cabin"].apply(get_deck)

deck_mapping = {"A": 1,
                "B": 2,
                "C": 3,
                "D": 4,
                "E": 5,
                "F": 6,
                "G": 7,
                "T": 8,
                "unknown": 0}

for k, v in deck_mapping.items():
    train_df["Deck"][train_df["Deck"] == k] = v
    test_df["Deck"][test_df["Deck"] == k] = v

# perform feature selection
predictors = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked", "Title",
    "Irish", "Aristocrat", "FamilyMembers", "FamilyWomen", "FamilyChildren", "WomanWChild",
              "FamilyCost", "Deck"]
selector = SelectKBest(f_classif, k=5)
selector.fit(train_df[predictors], train_df["Survived"])
# Get the raw p-values for each feature, and transform from p-values into scores
scores = -np.log10(selector.pvalues_)

# review predictive power of predictors
#plt.bar(range(len(predictors)), scores)
#plt.xticks(range(len(predictors)), predictors, rotation='vertical')
#plt.show()
predictors = ["Pclass", "Sex", "Embarked", "Title", "Irish", "FamilyWomen", "Fare", "Parch", "Age", "WomanWChild",
              "FamilyCost", "Deck"]

X_train, X_test, y_train, y_test = cross_validation.train_test_split(train_df[predictors], train_df["Survived"], test_size=0.3, random_state=42)

model = RandomForestRegressor(n_estimators=100, oob_score=True, random_state=42)
model = model.fit(X_train, y_train)
feature_importances = pd.Series(model.feature_importances_, index=X_train.columns)
feature_importances.sort()

predictions = model.predict(X_test)
accuracy = np.mean((predictions > 0.5).astype(int) == y_test)

print("c-stat:", roc_auc_score(y_train, model.oob_prediction_))
print("accuracy:", accuracy)
print("feature importance\n", feature_importances)


predictions = (model.predict(test_df[predictors]) > 0.5).astype(int)

# Kaggle needs the submission to have a certain format;
# see https://www.kaggle.com/c/titanic-gettingStarted/download/gendermodel.csv
# for an example of what it's supposed to look like.
submission = pd.DataFrame({ 'PassengerId': test_df['PassengerId'],
                           'Survived': predictions.astype(int) })
submission.to_csv("submission.csv", index=False)