# -*- coding: utf-8 -*-
"""
Created on Thu Aug  9 18:24:33 2018

@author: Yuangang
"""
import numpy as np
from zoopt.dimension import Dimension


class SemanticProblem(object):
    # 初始化函数
    def __init__(self, afs, train_labels, class_value):
        self.afs = afs
        self.train_labels = train_labels
        self.class_value = class_value

    # 将个体编码转换为复杂概念
    def code_to_complex_concept(self, solution):
        complex_concept = set()
        for i in range(0, len(solution)):
            if solution[i] == 0:
                continue
            else:
                complex_concept.add((i, solution[i] - 1))
        return complex_concept

        # 获取类内样本和类外样本标号

    def get_in_out_sample_index(self):
        in_class_sample_index = []
        out_class_sample_index = []
        for i in range(0, self.afs.data.shape[0]):
            if self.train_labels[i] == self.class_value:
                in_class_sample_index.append(i)
            else:
                out_class_sample_index.append(i)
        return in_class_sample_index, out_class_sample_index

    # 计算类内和类外样本的隶属度
    def calculate_in_out_mf(self, in_class_sample_index, out_class_sample_index, complex_concept):
        # 计算类内样本的隶属度
        in_mf = []
        for sample_index in in_class_sample_index:
            in_mf.append(self.afs.get_membership_degree_and(sample_index, complex_concept))

        # 计算类外样本的隶属度
        out_mf = []
        for sample_index in out_class_sample_index:
            out_mf.append(self.afs.get_membership_degree_and(sample_index, complex_concept))
        return in_mf, out_mf

    # 基于语义差的目标函数（全局）
    def fitness_semantic_global(self, solution):
        semantic_difference = 0.0
        x = solution.get_x()
        if x.count(0) == len(x):
            semantic_difference = 0.0
        else:
            in_class_sample_index, out_class_sample_index = self.get_in_out_sample_index()
            complex_concept = self.code_to_complex_concept(x)
            in_mf, out_mf = self.calculate_in_out_mf(in_class_sample_index, out_class_sample_index, complex_concept)
            semantic_difference = (sum(out_mf) / len(out_mf)) - (sum(in_mf) / len(in_mf))
        print("+++++++++++++++++++++++++++++++++++++++++++++++++")
        print(x)
        print(semantic_difference)
        return semantic_difference

    # 基于语义差的目标函数（局部）
    def fitness_semantic_local(self, solution):
        fitness_value = 0.0
        x = solution.get_x()
        if x.count(0) == len(x):
            fitness_value = 0.0
        else:
            in_class_sample_index, out_class_sample_index = self.get_in_out_sample_index()
            complex_concept = self.code_to_complex_concept(x)
            in_mf, out_mf = self.calculate_in_out_mf(in_class_sample_index, out_class_sample_index, complex_concept)
            fitness_value = max(out_mf) - max(in_mf)
        print("+++++++++++++++++++++++++++++++++++++++++++++++++")
        print(x)
        print(fitness_value)
        return fitness_value

    @property
    def dim(self):
        dimension_size = self.afs.data.shape[1]
        dimension_regions = [[0, self.afs.number_simple_concept]] * dimension_size
        #        dimension_regions = [list(range(0, self.afs.number_simple_concept + 1))] * dimension_size
        #        dimension_regions = []
        #        for i in range(0, dimension_size):
        #            region = [0, self.afs.number_simple_concept[i]]
        #            dimension_regions.append(region)
        dimension_types = [False] * dimension_size
        dimension_orders = [False] * dimension_size
        return Dimension(dimension_size, dimension_regions, dimension_types, dimension_orders)


class ErrorProblem(object):
    # 初始化函数
    def __init__(self, afs, true_label):
        self.afs = afs
        self.true_label = true_label

    # 将个体编码转换为复杂概念
    def individual_to_class_description(self, solution):
        class_description = []
        class_values = np.unique(self.true_label)
        for i in range(0, len(class_values)):
            complex_concept = set()
            for j in range(0, self.afs.data.shape[1]):
                si = i * self.afs.data.shape[1] + j
                if solution[si] == 0:
                    continue
                else:
                    complex_concept.add((j, solution[si] - 1))
            class_description.append(complex_concept)
        return class_description

    # 误差目标函数
    def fitness_error(self, solution):
        solution = solution.get_x()
        print(solution)
        class_description = self.individual_to_class_description(solution)
        predict_label = []
        for i in range(0, self.afs.data.shape[0]):
            mf_test = []
            for j in range(0, len(class_description)):
                mf_test.append(self.afs.get_membership_degree_and(i, class_description[j]))
            predict_label.append(mf_test.index(max(mf_test)))
        error_rate = 0
        for i in range(0, len(predict_label)):
            if self.true_label[i] == predict_label[i]:
                continue;
            else:
                error_rate = error_rate + 1
        error_rate = error_rate / len(predict_label)
        print(error_rate)
        print("==================================================")
        return error_rate

    @property
    def dim(self):
        dimension_size = self.afs.data.shape[1] * len(np.unique(self.true_label))
        # dimension_regions = [list(range(0, self.afs.number_simple_concept + 1))] * dimension_size
        dimension_regions = [[0, self.afs.number_simple_concept]] * dimension_size
        dimension_types = [False] * dimension_size
        dimension_orders = [False] * dimension_size
        return Dimension(dimension_size, dimension_regions, dimension_types, dimension_orders)
