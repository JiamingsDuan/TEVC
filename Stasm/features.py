# -*- coding: utf-8 -*-
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# 读取csv表格数据文件
score_csv_path = 'csv_data/zong_dawu_529.csv'
score = pd.read_csv(score_csv_path)[['N', 'E', 'O', 'A', 'C']]
landmark_csv_path = 'csv_data/Landmark_529.csv'
dataset = pd.read_csv(landmark_csv_path)
rows = dataset.shape[0]


# 计算任意多边形的面积，顶点按照顺时针或者逆时针方向排列
def compute_polygon_area(points):
    point_num = len(points)
    if point_num < 3:
        return 0.0
    s = points[0][1] * (points[point_num - 1][0] - points[1][0])

    for i in range(1, point_num):
        s += points[i][1] * (points[i - 1][0] - points[(i + 1) % point_num][0])
    return abs(s / 2.0)


L_eyebrow_list = []
R_eyebrow_list = []
L_eye_list = []
R_eye_list = []
nose_list = []
mouse_list = []
face_list = []
jaw_list = []

for row in range(0, rows):
    stu_landmark = list(dataset.loc[row, :])
    landmark_x = stu_landmark[1:154:2]
    landmark_y = stu_landmark[2:155:2]


    def generate_point(x0, y0):
        point = [landmark_x[x0], landmark_y[y0]]
        return point

    # 画出多边形的轮廓
    def generate_outline(start, close):
        point_list = []
        for x in range(start, close + 1):
            pointer = generate_point(x0=x, y0=x)
            point_list.append(pointer)
        square = compute_polygon_area(points=point_list)
        return square

    L_eyebrow = generate_outline(start=16, close=21)
    R_eyebrow = generate_outline(start=22, close=27)
    L_eye = generate_outline(30, 37)
    R_eye = generate_outline(40, 47)
    nose_up = generate_outline(51, 58)
    nose_down = generate_outline(48, 53)
    nose = nose_up + nose_down
    mouse_up = generate_outline(59, 68)
    mouse_down = generate_outline(69, 76)
    mouse = mouse_up + mouse_down
    face = generate_outline(0, 15)
    jaw = generate_outline(3, 9)

    L_eyebrow_list.append(L_eyebrow)
    R_eyebrow_list.append(R_eyebrow)
    L_eye_list.append(L_eye)
    R_eye_list.append(R_eye)
    nose_list.append(nose)
    mouse_list.append(mouse)
    face_list.append(face)
    jaw_list.append(jaw)


# 初始化
stu_numbers = list(dataset.iloc[:, 0])
stu_index = list(range(0, rows))
empty_frame = pd.DataFrame(index=stu_index)

empty_frame['NO'] = stu_numbers
empty_frame['L_eyebrow'] = L_eyebrow_list
empty_frame['R_eyebrow'] = R_eyebrow_list
empty_frame['L_eye'] = L_eye_list
empty_frame['R_eye'] = R_eye_list
empty_frame['nose'] = nose_list
empty_frame['mouse'] = mouse_list
empty_frame['face'] = face_list
empty_frame['jaw'] = jaw_list
# round(a/b,2)
empty_frame['eyebrow/face'] = (empty_frame['L_eyebrow'] + empty_frame['R_eyebrow']) / empty_frame['face']
empty_frame['eye/face'] = (empty_frame['L_eye'] + empty_frame['R_eye']) / empty_frame['face']
empty_frame['nose/face'] = empty_frame['nose'] / empty_frame['face']
empty_frame['mouse/face'] = empty_frame['mouse'] / empty_frame['face']
empty_frame['jaw/face'] = empty_frame['jaw'] / empty_frame['face']
# 取后四列
dataset_area = empty_frame.iloc[:, -5:]
trait_list = []
for row in range(0, rows):
    five_trait = list(score.iloc[row, :])
    trait_index = five_trait.index(max(five_trait)) + 1
    trait_list.append(trait_index)

dataset_area.insert(0, 'T', value=trait_list)
# 保存数据
# new_frame.to_csv('./csv_data/area_529.csv', sep=',', header=True, encoding='utf-8', index=False, )

# 【合并 start】
# dist_path = 'csv_data/Distance_1.csv'
# area_path = 'csv_data/area_529.csv'
#
# dist_data = pd.read_csv(dist_path)
# area_data = pd.read_csv(area_path)
#
# attributes = pd.concat([dist_data, area_data], axis=1)
# attributes.to_csv('csv_data/dist_area.csv', index=False, float_format='%.4f')
# 【合并 finish】


cols = dataset_area.shape[1]
attribute = dataset_area.iloc[:, 1:cols]
labels = dataset_area.iloc[:, 0]
X1 = attribute.values
y = labels.values


for k in range(1, cols):
    selector = SelectKBest(score_func=f_classif, k=k)
    selector.fit(X1, y)
    select_best_index = selector.get_support(True)
    X = attribute.iloc[:, select_best_index].values
    if k == 3:
        print(select_best_index)
    else:
        pass
    # split the dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)

    classifier = RandomForestClassifier(n_estimators=100, criterion='entropy', random_state=0)
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)
    score = accuracy_score(y_test, y_pred)
    print('accurate:', '%.2f' % score, 'feature_num:k=', k)
