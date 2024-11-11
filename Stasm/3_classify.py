import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler

data_path = 'csv_data/features_28.csv'
score_path = 'csv_data/zong_dawu_529.csv'
Dataset = pd.read_csv(data_path)
Scores = pd.read_csv(score_path)[['N', 'E', 'O', 'A', 'C']]

Neu_data = pd.concat([Dataset, Scores['N']], axis=1)
Ext_data = pd.concat([Dataset, Scores['E']], axis=1)
Open_data = pd.concat([Dataset, Scores['O']], axis=1)
Agr_data = pd.concat([Dataset, Scores['A']], axis=1)
Con_data = pd.concat([Dataset, Scores['C']], axis=1)

Scores_array = Scores.values
Scale = MinMaxScaler()
Score_normalized = Scale.fit_transform(Scores_array)
col = Dataset.shape[1] + 1
Neu_data.insert(col, 'n', value=Score_normalized[:, 0])
Ext_data.insert(col, 'e', value=Score_normalized[:, 1])
Open_data.insert(col, 'o', value=Score_normalized[:, 2])
Agr_data.insert(col, 'a', value=Score_normalized[:, 3])
Con_data.insert(col, 'c', value=Score_normalized[:, 4])

plt.hist(Score_normalized[:, 0], bins=list(np.arange(0.0, 1.1, 0.1)))
plt.scatter(x=list(range(1, 530)), y=Neu_data['n'].sort_values())
plt.ylim(0, 1)
plt.show()


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

    rows_1 = data[data[trait] < x1].shape[0]
    rows_2 = data[(data[trait] > x1) & (data[trait] < x2)].shape[0]
    rows_3 = data[data[trait] > x2].shape[0]
    print(rows_1, rows_2, rows_3)
    return label


features = Neu_data.iloc[:, :-2]
labels = partition(Ext_data, 0.4, 0.6, 'e')
X = features.values
y = labels

for step in range(5, 29):
    selector = SelectKBest(score_func=f_classif, k=step)
    selector.fit(X, y)
    select_best_index = selector.get_support(True)
    X0 = features.iloc[:, select_best_index].values

    X_train, X_test, y_train, y_test = train_test_split(X0, y, test_size=0.4, random_state=0)
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)
    classifier = RandomForestClassifier(n_estimators=200, criterion='entropy', random_state=0)
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)
    accurate = accuracy_score(y_test, y_pred)
    print('accurate:', '%.2f' % accurate, 'feature_num:k=', step)
