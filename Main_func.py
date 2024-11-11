import math
import numpy as np

from dcfs.dcfs import DCFS


def generate_time_series():
    # 生成Mackey-Glass时间序列 time-series plus noise
    p = []  # list length = 3100
    for i in range(0, 50):
        p.append(0.04 * i)

    r = []  # list length = 3050
    for i in range(50, 3100):
        p.append(0.9 * p[i - 1] + 0.2 * p[i - 50] / (1 + p[i - 50] ** 10))
        r.append(math.log(p[i] / p[i - 1]) + 0.0001 * np.random.randn())
    return r


# 生成时间序列
Series = generate_time_series()


def save_series(arr):
    with open('series/Time_Series.txt', 'w') as f:
        for item in list(arr):
            f.write(str(item))
            f.write(',')
        f.close()


# 保存数据集
save_series(Series)


# 生成数据表
def generate_data_table(t, arr):
    data = np.zeros((3000, t))  # 生成3000行t列的由0.(默认为浮点型)填充的矩阵
    for i in range(0, 3000):
        for j in range(0, t):
            data[i, j] = arr[i + j]
    return data
