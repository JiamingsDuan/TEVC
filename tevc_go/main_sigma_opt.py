# -*- coding: utf-8 -*-
"""
Created on Tue Jun  4 14:22:14 2019

@author: kbawy
"""

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score
from zoopt import Objective, Parameter, Opt

from afs import AFS, LogicOperation
from problem import SemanticProblem, ErrorProblem

# 加载训练数据
train_dataset = np.loadtxt('dataset/10fold/haberman/haberman-10-8tra.dat', delimiter=',')
train_data_index = []
for i in range(0, train_dataset.shape[1] - 1):
    train_data_index.append(i)
train_data = train_dataset[:, train_data_index]
train_target = train_dataset[:, train_dataset.shape[1] - 1]
train_t = np.ones(len(train_target))
train_label = train_target - train_t

# 加载测试数据
test_dataset = np.loadtxt('dataset/10fold/haberman/haberman-10-8tst.dat', delimiter=',')
test_data_index = []
for i in range(0, test_dataset.shape[1] - 1):
    test_data_index.append(i)
test_data = test_dataset[:, test_data_index]
test_target = test_dataset[:, test_dataset.shape[1] - 1]
test_t = np.ones(len(test_target))
test_label = test_target - test_t

# 算法参数配置
# 参数优化配置
delta = 0.1
iterations = 30

# AFS算法配置
logic_index_and = LogicOperation.product
logic_index_or = LogicOperation.maximum
number_simple_concept = 3
gaussian_sigma = 5.0

# RACOS算法配置
# 基于语义差
train_size_semantic = 6
positive_size_semantic = 4
negative_size_semantic = 2

# 基于训练集的错误率
train_size_error = 10
positive_size_error = 6
negative_size_error = 4

accuracy_iterations = {}

for iteration in range(0, iterations):
    # 训练分类器
    # 构建AFS框架
    afs = AFS(train_data, logic_index_and, logic_index_or, number_simple_concept, gaussian_sigma + delta)
    afs.build_afs()
    print(afs.simple_concepts)

    # 获取类标签
    class_values = np.unique(train_label)

    # 生成类别的描述
    class_description_semantic_global = []
    class_description_semantic_local = []

    # 基于语义差的类别描述生成
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
        class_description_semantic_global.append(
            problem_semantic.code_to_complex_concept(solution_semantic_global.get_x()))
        class_description_semantic_local.append(
            problem_semantic.code_to_complex_concept(solution_semantic_local.get_x()))

    # 基于训练集上错误率的类别描述生成
    problem_error = ErrorProblem(afs, train_label)
    parameter_error = Parameter(budget=100 * problem_semantic.dim.get_size(), autoset=False)
    parameter_error.set_train_size(train_size_error)
    parameter_error.set_positive_size(positive_size_error)
    parameter_error.set_negative_size(negative_size_error)
    objective_error = Objective(problem_error.fitness_error, problem_error.dim)
    solution_error = Opt.min(objective_error, parameter_error)
    class_description_error = problem_error.individual_to_class_description(solution_error.get_x())

    # 利用or将两种描述连接形成类别的最终描述
    class_description = []
    for i in range(0, len(class_values)):
        des = []
        des.append(class_description_semantic_global[i])
        des.append(class_description_semantic_local[i])
        des.append(class_description_error[i])
        class_description.append(des)

    # 绘制图像
    class_names = ['Class1', 'Class2', 'Class3']
    # print (class_description_error)

    plt.figure()
    plt.style.use('ggplot')
    colors = ['#DD3429', '#139174', '#2E4075']
    markers = ['o', 's', '^']

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
        plt.plot(degree_ordered, marker=markers[index], linewidth=2, label=class_names[index])
    plt.legend()
    plt.show()

    #    plt.figure()
    #    plt.style.use('ggplot')
    #    plt.plot(objective_semantic_global.get_history_bestsofar())
    #    plt.show()
    #
    #    plt.figure()
    #    plt.style.use('ggplot')
    #    plt.plot(objective_semantic_local.get_history_bestsofar())
    #    plt.show()

    # plt.figure()
    # plt.style.use('ggplot')
    # plt.plot(objective_error.get_history_bestsofar())
    # plt.show()

    # 利用类别的语义描述分类测试样本
    predict_label = []
    for i in test_data:
        train_test_data = np.row_stack((train_data, i))
        afs_test = AFS(train_test_data, logic_index_and, logic_index_or, number_simple_concept, afs.gaussian_sigma)
        afs_test.build_afs()
        mf_test = []
        for j in range(0, len(class_values)):
            mf_test.append(afs_test.get_membership_degree_and_or(train_test_data.shape[0] - 1, class_description[j]))
        predict_label.append(mf_test.index(max(mf_test)))

    accuracy_iterations.setdefault(afs.gaussian_sigma, accuracy_score(test_label, predict_label))
    gaussian_sigma = afs.gaussian_sigma

accuracy_iterations = sorted(accuracy_iterations.items(), key=lambda d: d[0], reverse=False)

print(accuracy_iterations)
