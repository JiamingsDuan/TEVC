#  -*- coding: utf-8 -*-
"""
Created on Mon May 27 14:19:47 2019

@author: kbawy
"""

#  主函数：10fold交叉验证

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score
#  使用的包
from sklearn.model_selection import train_test_split
from zoopt import Objective, Parameter, Opt

from afs import AFS, LogicOperation
from problem import SemanticProblem, ErrorProblem

#  1.从文件中读取数据
dataset = np.loadtxt('./data/face-3.arff', delimiter=',')

#  2.整理数据
data_index = []
for i in range(0, dataset.shape[1] - 1):
    data_index.append(i)
data = dataset[:, data_index]
target = dataset[:, dataset.shape[1] - 1]
# target = [i - 1 for i in target]
t = np.ones(len(target))
target = target - t
print(target)

#  3.算法参数配置
# 3.1 AFS算法配置
logic_index_and = LogicOperation.product
logic_index_or = LogicOperation.maximum
number_simple_concept = 3
gaussian_sigma = 0.1

# 3.2 RACOS算法配置
# 3.2.1 基于语义差
train_size_semantic = 6
positive_size_semantic = 4
negative_size_semantic = 2

# 3.2.2 基于训练集的错误率
train_size_error = 10
positive_size_error = 6
negative_size_error = 4

# 4. 将数据集分割为训练集和测试集（10%测试，90%训练）
train_data, test_data, train_label, test_label = train_test_split(data, target, test_size=0.1, random_state=12)

# 5. 训练分类器
# 5.1 构建AFS框架
afs = AFS(train_data, logic_index_and, logic_index_or, number_simple_concept, gaussian_sigma)
afs.build_afs()
print(afs.simple_concepts)

# 5.2 获取类标签
class_values = np.unique(train_label)

# 5.3 生成类别的描述
class_description_semantic_global = []
class_description_semantic_local = []

# 5.3.1 基于语义差的类别描述生成
for class_value in class_values:
    # 定义要优化的问题
    problem_semantic = SemanticProblem(afs, train_label, class_value)

    # 配置优化算法的参数
    parameter_semantic = Parameter(budget=100 * problem_semantic.dim.get_size(), autoset=False)
    parameter_semantic.set_train_size(train_size_semantic)
    parameter_semantic.set_positive_size(positive_size_semantic)
    parameter_semantic.set_negative_size(negative_size_semantic)

    # 配置优化算法的目标函数
    objective_semantic_global = Objective(problem_semantic.fitness_semantic_global, problem_semantic.dim)
    objective_semantic_local = Objective(problem_semantic.fitness_semantic_local, problem_semantic.dim)

    # 利用优化算法得到最优解
    solution_semantic_global = Opt.min(objective_semantic_global, parameter_semantic)
    solution_semantic_local = Opt.min(objective_semantic_local, parameter_semantic)

    # 类别描述
    class_description_semantic_global.append(problem_semantic.code_to_complex_concept(solution_semantic_global.get_x()))
    class_description_semantic_local.append(problem_semantic.code_to_complex_concept(solution_semantic_local.get_x()))
    print(class_description_semantic_local)
    print(class_description_semantic_global)

# 5.3.2 基于训练集上错误率的类别描述生成
problem_error = ErrorProblem(afs, train_label)
parameter_error = Parameter(budget=100 * problem_semantic.dim.get_size(), autoset=False)
parameter_error.set_train_size(train_size_error)
parameter_error.set_positive_size(positive_size_error)
parameter_error.set_negative_size(negative_size_error)
objective_error = Objective(problem_error.fitness_error, problem_error.dim)
solution_error = Opt.min(objective_error, parameter_error)
class_description_error = problem_error.individual_to_class_description(solution_error.get_x())

# 5.3.3 利用or将两种描述连接形成类别的最终描述
class_description = []
for i in range(0, len(class_values)):
    des = [class_description_semantic_global[i], class_description_semantic_local[i], class_description_error[i]]
    class_description.append(des)

# 绘制图像
# class_names = ['Class1', 'Class2', 'Class3']
# print (class_description_error)

# plt.figure()
# plt.style.use('ggplot')
# colors = ['# DD3429', '# 139174', '# 2E4075']
# markers = ['o', 's', '^']

class_values = np.unique(train_label)

for index, value in enumerate(class_description):
    degree = afs.get_membership_function(value)
    # 将degree根据类标签group
    degree_ordered = []
    for i in range(0, len(class_values)):
        print(type(train_label))
        same_class_sample_indexes = np.where(train_label == class_values[i])
        for j in same_class_sample_indexes[0]:
            degree_ordered.append(degree[j])
    # plt.plot(degree_ordered, marker=markers[index], linewidth=2, label=class_names[index])
# plt.legend()
# plt.show()

# plt.figure()
# plt.style.use('ggplot')
# plt.plot(objective_semantic_global.get_history_bestsofar())
# plt.show()

# plt.figure()
# plt.style.use('ggplot')
# plt.plot(objective_semantic_local.get_history_bestsofar())
# plt.show()

# plt.figure()
# plt.style.use('ggplot')
# plt.plot(objective_error.get_history_bestsofar())
# plt.show()


# 3.利用类别的语义描述分类测试样本

predict_label = []
for i in test_data:
    train_test_data = np.row_stack((train_data, i))
    afs_test = AFS(train_test_data, logic_index_and, logic_index_or, number_simple_concept, gaussian_sigma)
    afs_test.build_afs()
    mf_test = []
    for j in range(0, len(class_values)):
        mf_test.append(afs_test.get_membership_degree_and_or(train_test_data.shape[0] - 1, class_description[j]))
    predict_label.append(mf_test.index(max(mf_test)))

# print(class_description_semantic_global)
# print(class_description_semantic_local)
print(accuracy_score(test_label, predict_label))
