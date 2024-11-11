# -*- coding: utf-8 -*-
"""
Created on Mon Dec 11 17:14:05 2017

@author: Yuangang Wang
"""
import math
from enum import Enum

import numpy as np


class LogicOperation(Enum):
    product = 1
    average = 2
    maximum = 3
    minimum = 4


class AFS(object):

    def __init__(self, data, logic_operation_and, logic_operation_or, number_simple_concept, gaussian_sigma):
        self.data = data
        self.logic_operation_and = logic_operation_and
        self.logic_operation_or = logic_operation_or
        self.gaussian_sigma = gaussian_sigma
        self.number_simple_concept = number_simple_concept
        self.simple_concepts = {}
        self.weight = {}
        self.afs_structure = {}
        return

    def build_afs(self):
        self.generate_simple_concepts()
        print('Finish generating simple concepts...')
        self.generate_afs_structure()
        print('Finish generating AFS structure...')

    def generate_simple_concepts(self):
        for i in range(self.data.shape[1]):
            feature_values = self.data[:, i]
            temp_max = np.amax(feature_values)
            temp_min = np.amin(feature_values)
            temp_interval = (temp_max - temp_min) / (self.number_simple_concept - 1)
            for j in range(self.number_simple_concept):
                self.simple_concepts.update({(i, j): ((temp_min + j * temp_interval), self.gaussian_sigma)})
        return

    def generate_afs_structure(self):
        for i in range(self.data.shape[0]):
            for c in self.simple_concepts.keys():
                self.weight.setdefault(('sum', c), 0.0)
                temp_weight = self.gaussian_weight_function(self.data[i, c[0]], self.simple_concepts.get(c)[0],
                                                            self.simple_concepts.get(c)[1])
                self.weight.update({(i, c): temp_weight})
                self.weight.update({('sum', c): (self.weight.get(('sum', c)) + temp_weight)})
                for j in range(self.data.shape[0]):
                    self.afs_structure.setdefault((i, c), set())
                    if abs(self.data[i, c[0]] - self.simple_concepts.get(c)[0]) <= abs(
                            self.data[j, c[0]] - self.simple_concepts.get(c)[0]):
                        self.afs_structure.get((i, c)).add(j)
        return

    def gaussian_weight_function(self, x, mu, sigma):
        return math.exp(-0.5 * (math.pow((x - mu), 2) / math.pow(sigma, 2)))

    def get_membership_degree_and(self, sample_index, concept):
        degree_simple_concepts = []
        for c in concept:
            covered_samples = self.afs_structure.get((sample_index, c))
            temp_sum = 0.0
            for i in covered_samples:
                temp_sum = temp_sum + self.weight.get((i, c))
            degree_simple_concepts.append(temp_sum / (self.weight.get(('sum', c)) + 0.0000000001))
        if self.logic_operation_and == LogicOperation.product:
            return np.prod(degree_simple_concepts)
        elif self.logic_operation_and == LogicOperation.average:
            return np.mean(degree_simple_concepts)
        elif self.logic_operation_and == LogicOperation.maximum:
            return np.max(degree_simple_concepts)
        elif self.logic_operation_and == LogicOperation.minimum:
            return np.min(degree_simple_concepts)

    def get_membership_degree_and_or(self, sample_index, concept):
        degree_complex_concepts = []
        for c in concept:
            degree_complex_concepts.append(self.get_membership_degree_and(sample_index, c))
        if self.logic_operation_or == LogicOperation.average:
            return np.mean(degree_complex_concepts)
        elif self.logic_operation_or == LogicOperation.maximum:
            return np.max(degree_complex_concepts)
        elif self.logic_operation_or == LogicOperation.minimum:
            return np.min(degree_complex_concepts)

    def get_membership_function(self, concept):
        degrees = []
        for i in range(self.data.shape[0]):
            degrees.append(self.get_membership_degree_and_or(i, concept))
        return degrees
