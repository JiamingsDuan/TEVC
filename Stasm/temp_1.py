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

plt.hist(Score_normalized[:, 0], bins=20)
# plt.scatter(x=list(range(1, 530)), y=Features['N'])
plt.show()
