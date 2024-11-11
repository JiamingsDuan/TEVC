import os
import numpy as np
from pandas import DataFrame
import pandas as pd

# 照片目录
landmarks_path = "./data/landmark_2014/"
# 点名称文件目录
landmark_name_path = 'csv_data/landmarks_name.txt'


def make_frame(landmarks, names):
    # 存储学生学号的列表
    student_num = []
    for file in os.listdir(landmarks):
        file_name = os.path.splitext(file)[0]
        student_num.append(str(file_name))

    # 存储点坐标文件名的列表
    file_name_list = os.listdir(landmarks)

    # 存储每个点名称的列表
    name_list = []
    with open(names, 'r') as ff:
        lines = ff.read().splitlines()
        ff.close()
    for name in lines:
        name_list.append(str(name))

    # 点坐标表
    feature_table = []
    for f in file_name_list:
        file_name = landmarks + f
        lines = open(file_name).readlines()
        with open(file_name, 'w') as fp:
            for s in lines:
                fp.write(s.replace(' ', ''))
            fp.close()

        with open(file_name, 'w') as fp:
            for s in lines:
                fp.write(s.replace(',', '\n'))
            fp.close()

        with open(file_name, 'r') as fp:
            lines = fp.read().splitlines()
            fp.close()

        feature_list = []
        for i in lines:
            feature_list.append(int(i))
        # print(feature_list)
        # print(len(feature_list))
        # landmark_X = feature_list[0:153:2]  # 横坐标
        # landmark_Y = feature_list[1:154:2]  # 纵坐标
        feature_table.append(feature_list)

    # list 转 array
    landmarks_array = np.array(feature_table)
    # print(np.shape(landmarks_array))  # (94, 154)
    # 用pandas转换成frame格式的数据表
    landmarks_frame = DataFrame(landmarks_array, columns=name_list)
    # 在第一列添加学号列
    landmarks_frame.insert(0, 'No', student_num)
    return landmarks_frame


landmark_frame = make_frame(landmarks=landmarks_path, names=landmark_name_path)
landmark_frame.to_csv('landmarks_2014.csv', sep=',', header=True, encoding='utf-8', index=False)

landmark_csv_path = 'csv_data/landmarks_2014.csv'
landmark_frames = pd.read_csv(landmark_csv_path)
