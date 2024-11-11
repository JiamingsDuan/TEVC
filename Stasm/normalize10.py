import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter

data_path = 'csv_data/zong_dawu_529.csv'
dataset = pd.read_csv(data_path)[['性别', 'N', 'E', 'O', 'A', 'C']]
# 行数
rows = dataset.shape[0]
m_dataset = dataset.drop(index=dataset.loc[(dataset['性别'] == '男')].index)[['N', 'E', 'O', 'A', 'C']]
f_dataset = dataset.drop(index=dataset.loc[(dataset['性别'] == '女')].index)[['N', 'E', 'O', 'A', 'C']]
# calculate std mean
m_mean = list(m_dataset.describe().iloc[1])
f_mean = list(f_dataset.describe().iloc[1])
m_std = list(m_dataset.describe().iloc[2])
f_std = list(f_dataset.describe().iloc[2])


def pointer(score, mean, std):
    mark = 0
    if score < mean - 2 * std:
        mark = 1
    elif mean - 2 * std <= score < mean - 1.5 * std:
        mark = 2
    elif mean - 1.5 * std <= score < mean - std:
        mark = 3
    elif mean - std <= score < mean - 0.5 * std:
        mark = 4
    elif mean - 0.5 * std <= score < mean:
        mark = 5
    elif mean < score <= mean + 0.3 * std:
        mark = 6
    elif mean + 0.3 * std <= score < mean + 0.6 * std:
        mark = 7
    elif mean + 0.6 * std <= score < mean + std:
        mark = 8
    elif mean + std <= score < mean + 1.5 * std:
        mark = 9
    elif score >= mean + 1.5 * std:
        mark = 10
    else:
        pass
    return mark


def generate_label(mark):
    label = 0
    if mark <= 3:
        label = 1
    elif 3 < mark < 7:
        label = 2
    elif mark >= 7:
        label = 3
    else:
        pass
    return label


# 转十分制
for row in range(0, rows):
    score6 = list(dataset.iloc[row])
    if score6[0] == '男':
        score6[1] = pointer(score6[1], mean=m_mean[0], std=m_std[0])
        score6[2] = pointer(score6[2], mean=m_mean[1], std=m_std[1])
        score6[3] = pointer(score6[3], mean=m_mean[2], std=m_std[2])
        score6[4] = pointer(score6[4], mean=m_mean[3], std=m_std[3])
        score6[5] = pointer(score6[5], mean=m_mean[4], std=m_std[4])
    elif score6[0] == '女':
        score6[1] = pointer(score6[1], mean=f_mean[0], std=f_std[0])
        score6[2] = pointer(score6[2], mean=f_mean[1], std=f_std[1])
        score6[3] = pointer(score6[3], mean=f_mean[2], std=f_std[2])
        score6[4] = pointer(score6[4], mean=f_mean[3], std=f_std[3])
        score6[5] = pointer(score6[5], mean=f_mean[4], std=f_std[4])
    else:
        pass
    dataset.iloc[row] = score6

system10 = dataset[['N', 'E', 'O', 'A', 'C']]

# 作图
fig = plt.figure()
ax1 = fig.add_subplot(321)
ax2 = fig.add_subplot(322)
ax3 = fig.add_subplot(323)
ax4 = fig.add_subplot(324)
ax5 = fig.add_subplot(325)
ax1.hist(system10['N'], bins=10)
ax2.hist(system10['E'], bins=10)
ax3.hist(system10['O'], bins=10)
ax4.hist(system10['A'], bins=10)
ax5.hist(system10['C'], bins=10)
plt.show()

system3 = system10

for row1 in range(0, rows):
    score5 = list(system3.iloc[row1])
    score5[0] = generate_label(score5[0])
    score5[1] = generate_label(score5[1])
    score5[2] = generate_label(score5[2])
    score5[3] = generate_label(score5[3])
    score5[4] = generate_label(score5[4])
    system3.iloc[row1][0] = score5[0]
    system3.iloc[row1][1] = score5[1]
    system3.iloc[row1][2] = score5[2]
    system3.iloc[row1][3] = score5[3]
    system3.iloc[row1][4] = score5[4]

for i in system3.columns:
    amount = Counter(list(system3[i]))
    print(i, ':', amount)
system3.insert(0, '学号', value=list(pd.read_csv(data_path)['学号']))
system3.to_csv('csv_data/label_529.csv', sep=',', encoding='utf-8', index=False)
