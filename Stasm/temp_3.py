# -*- coding: utf-8 -*-
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.preprocessing import normalize

data_path = 'csv_data/features_28.csv'
score_path = 'csv_data/zong_dawu_529.csv'
Data = pd.read_csv(data_path)
Scores = pd.read_csv(score_path)[['N', 'E', 'O', 'A', 'C']]

# score Normalized
Scores_value = Scores.values
Scale = MinMaxScaler()
Score_normalized = Scale.fit_transform(Scores_value)

col = Data.shape[1]
Data.insert(col, 'N', value=Score_normalized[:, 0])
Data.insert(col + 1, 'E', value=Score_normalized[:, 1])
Data.insert(col + 2, 'O', value=Score_normalized[:, 2])
Data.insert(col + 3, 'A', value=Score_normalized[:, 3])
Data.insert(col + 4, 'C', value=Score_normalized[:, 4])

# neuroticism
frame_neuroticism = Data[Data['T'] == 1]
# extroversion
frame_extroversion = Data[Data['T'] == 2]
# openness
frame_openness = Data[Data['T'] == 3]
# agreeableness
frame_agreeableness = Data[Data['T'] == 4]
# conscientiousness
frame_conscientiousness = Data[Data['T'] == 5]

plt.hist(frame_conscientiousness['C'], bins=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
plt.show()


# 一个性格特质的五个范围
def Subdivision(frame, trait):
    frame_1 = frame[frame[trait] < 0.2]
    frame_2 = frame[(frame[trait] < 0.4) & (frame[trait] > 0.2)]
    frame_3 = frame[(frame[trait] < 0.6) & (frame[trait] > 0.4)]
    frame_4 = frame[(frame[trait] < 0.8) & (frame[trait] > 0.6)]
    frame_5 = frame[frame[trait] > 0.8]
    sub_frame = [frame_1, frame_2, frame_3, frame_4, frame_5]
    return sub_frame


neuroticism_sub_frame = Subdivision(frame_neuroticism, trait='N')
extroversion_sub_frame = Subdivision(frame_extroversion, trait='E')
openness_sub_frame = Subdivision(frame_openness, trait='O')
agreeableness_sub_frame = Subdivision(frame_agreeableness, trait='A')
conscientiousness_sub_frame = Subdivision(frame_conscientiousness, trait='C')


def Partial_division(dataset):
    X0 = dataset.iloc[:, 1:col - 5]
    y0 = dataset.iloc[:, 0]
    X_train, X_test, y_train, y_test = train_test_split(X0, y0, test_size=0.25, random_state=0)
    train_data = pd.concat([X_train, y_train], axis=1)
    test_data = pd.concat([X_test, y_test], axis=1)
    return train_data, test_data


N_train_list = []
N_test_list = []
for i in neuroticism_sub_frame:
    train, test = Partial_division(i)
    N_train_list.append(train)
    N_test_list.append(test)

N_train_set = pd.concat(N_train_list, axis=0)
N_test_set = pd.concat(N_test_list, axis=0)
