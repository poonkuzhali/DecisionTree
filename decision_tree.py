import numpy
import numpy as np

from importance_formulas import (gain_ratio_categorical, gain_ratio_continuous, gini_index_categorical,
                                 gini_index_continuous, gain_continuous)


def remove_attr(attr, attr_list):
    return [a for a in attr_list if a != attr]


def predict(tree, example):
    if isinstance(tree, BranchNode):
        attr_val = example[tree.attr_idx]
        return predict(tree.branches.get(attr_val, tree.node), example)
    elif isinstance(tree, LeafNode):
        return tree.value


def classify_popular_label(data):
    label_column = data[:, -1]
    unique_classes, counts = np.unique(label_column, return_counts=True)
    classification = unique_classes[counts.argmax()]
    return classification


def isFeatureCategorical(dataset, idx):
    data_type = type(dataset[:, idx][0])
    if data_type == np.object_ or data_type == str:
        return True
    else:
        return False


def plurality_value(dataset):
    unique_labels, counts = np.unique(dataset[:, -1], return_counts=True)
    max_label = unique_labels[np.argmax(counts)]
    return LeafNode(max_label)


def same_classification(dataset):
    label = dataset[0, -1]
    for x in dataset:
        if x[-1] != label:
            return False
    return True


class LeafNode:
    def __init__(self, value):
        self.value = value


class BranchNode:
    def __init__(self, attr_idx, attr_name=None, node=None, branches=None):
        self.attr_idx = attr_idx
        self.attr_name = attr_name
        self.node = node
        self.branches = branches or {}

    def add(self, attr, subtree):
        self.branches[attr] = subtree


class DecisionTreeModel:
    def __init__(self, dataset, method="c4.5"):
        self.dataset = dataset
        self.method = method

    def decision_tree_learning(self, dataset, attributes=None, parent_examples=None):
        if len(dataset) == 0:
            return plurality_value(parent_examples)
        if same_classification(dataset):
            return LeafNode(dataset[0, -1])
        if len(attributes) == 0:
            return plurality_value(dataset)
        else:
            attr_idx = -1
            if self.method == "c4.5":
                max_info_gain = -float("inf")
                for i in range(len(attributes) - 1):
                    if isFeatureCategorical(dataset, i):
                        gain_ratio = gain_ratio_categorical(dataset, i)
                    else:
                        gain_ratio = gain_ratio_continuous(dataset, i)

                    if gain_ratio > max_info_gain:
                        max_info_gain = gain_ratio
                        attr_idx = i
            elif self.method == "cart":
                min_gini = float("inf")
                for i in range(len(attributes) - 1):
                    if isFeatureCategorical(dataset, i):
                        gini_index = gini_index_categorical(dataset, i)
                    else:
                        gini_index, t = gini_index_continuous(dataset, i)

                    if gini_index < min_gini:
                        min_gini = gini_index
                        attr_idx = i

            tree = BranchNode(attr_idx, attributes[attr_idx], plurality_value(dataset))
            if isFeatureCategorical(dataset, attr_idx):
                for value, examples in self.split_function_categorical(attr_idx, dataset).items():
                    examples = numpy.asarray(examples)
                    subtree = self.decision_tree_learning(examples, remove_attr(attributes[attr_idx], attributes),
                                                          dataset)
                    tree.add(value, subtree)
            else:
                for value, examples in self.split_function_continuous(attr_idx, dataset).items():
                    examples = numpy.asarray(examples)
                    subtree = self.decision_tree_learning(examples, remove_attr(attributes[attr_idx], attributes),
                                                          dataset)
                    tree.add(value, subtree)
            return tree

    def split_function_categorical(self, attr_idx, dataset):
        splits = {}
        column = dataset[:, attr_idx]
        unique_values = np.unique(column)

        for value in unique_values:
            examples = []
            for data in dataset:
                if data[attr_idx] == value:
                    examples.append(data)
            splits[value] = examples

        return splits

    def split_function_continuous(self, attr_idx, dataset):
        best_threshold = None

        if self.method == "c4.5":
            info_gain, best_threshold = gain_continuous(dataset, attr_idx)
        elif self.method == "cart":
            gini, best_threshold = gini_index_continuous(dataset, attr_idx)

        splits = {f"<= {best_threshold}": dataset[dataset[:, attr_idx] <= best_threshold],
                  f"> {best_threshold}": dataset[dataset[:, attr_idx] > best_threshold]}
        return splits
