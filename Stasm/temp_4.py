# -*- coding: utf-8 -*-
import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

data_path = 'csv_data/features_28.csv'
score_path = 'csv_data/zong_dawu_529.csv'
Features = pd.read_csv(data_path)
Scores = pd.read_csv(score_path)

score_data = Scores[['学号', 'N', 'E', 'O', 'A', 'C']]
Proofreading_data = pd.concat([score_data, Features], axis=1)

Score = Scores[['N', 'E', 'O', 'A', 'C']].values
Scale = MinMaxScaler()
Score_normalized = Scale.fit_transform(Score)

Features.insert(loc=1, column='N', value=Score_normalized[:, 0])
Features.insert(loc=2, column='E', value=Score_normalized[:, 1])
Features.insert(loc=3, column='O', value=Score_normalized[:, 2])
Features.insert(loc=4, column='A', value=Score_normalized[:, 3])
Features.insert(loc=5, column='C', value=Score_normalized[:, 4])
del Features['学号']

# Feature mean
describe_table = Features.describe()
mean_series = describe_table.iloc[1, -28:]

plt.hist(Score_normalized[:, 0], bins=list(np.arange(0.0, 1.1, 0.1)))
plt.scatter(x=list(range(1, 530)), y=Features['N'].sort_values())
plt.ylim(0, 1)
# plt.show()


def partition(data, x1, x2, trait):
    label = []
    score_list = list(data[trait])
    for val in score_list:
        if val < x1:
            val = 1
        elif x1 < val < x2:
            val = 2
        else:
            val = 3
        label.append(val)

    # Number of samples of the first
    rows_1 = data[data[trait] < x1].shape[0]
    # Number of samples of the second
    rows_2 = data[(data[trait] > x1) & (data[trait] < x2)].shape[0]
    # Number of samples of the last
    rows_3 = data[data[trait] > x2].shape[0]
    print('first group', rows_1, 'second group', rows_2, 'third group', rows_3)
    return label


# X: feature_set
features = Features.iloc[:, -28:]
label_name = ['N', 'E', 'O', 'A', 'C']
label = partition(data=Features, x1=0.33, x2=0.66, trait='A')


def generate_train_test(f):

    f.insert(0, 'L', value=label)
    X1 = f[f['L'] == 1].iloc[:, -28:]
    y1 = f[f['L'] == 1].iloc[:, 0]
    X1_train, X1_test, y1_train, y1_test = train_test_split(X1, y1, test_size=0.2, random_state=0)
    X2 = f[f['L'] == 2].iloc[:, -28:]
    y2 = f[f['L'] == 2].iloc[:, 0]
    X2_train, X2_test, y2_train, y2_test = train_test_split(X2, y2, test_size=0.2, random_state=0)
    X3 = f[f['L'] == 3].iloc[:, -28:]
    y3 = f[f['L'] == 3].iloc[:, 0]
    X3_train, X3_test, y3_train, y3_test = train_test_split(X3, y3, test_size=0.2, random_state=0)
    X0_train = pd.concat([X1_train, X2_train, X3_train], axis=0)
    X0_test = pd.concat([X1_test, X2_test, X3_test], axis=0)
    y0_train = pd.concat([y1_train, y2_train, y3_train], axis=0)
    y0_test = pd.concat([y1_test, y2_test, y3_test], axis=0)
    train_set = pd.concat([X0_train, y0_train], axis=1).sample(frac=1.0)
    test_set = pd.concat([X0_test, y0_test], axis=1).sample(frac=1.0)

    return train_set, test_set


train, test = generate_train_test(features)
X_train = train.iloc[:, : -1].values
y_train = train.iloc[:, -1:].values
X_test = test.iloc[:, : -1].values
y_test = test.iloc[:, -1:].values
# classifier = LogisticRegression(penalty='l2')
classifier = RandomForestClassifier(n_estimators=200, criterion='entropy', random_state=0)
classifier.fit(X_train, y_train.ravel())
y_predict = classifier.predict(X_test)
acc = accuracy_score(y_test, y_predict)
print('accurate:', '%.2f' % acc)
