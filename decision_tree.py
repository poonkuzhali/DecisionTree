import numpy as np


class DecisionTree:
    def __init__(self, branches=None):
        self.branches = branches or {}

    def isFeatureCategorical(self, dataset, idx):
        data_type = type(dataset[:, idx][0])
        if data_type == np.object_ or data_type == str:
            return True
        else:
            return False

    def decision_tree_learning(self, examples, attributes, parent_examples):
        pass

    def build_decision_tree(self):
        pass

    def best_split_fn(self, dataset, n_attr):
        best_split = {}
        max_info_gain = -float("inf")

        for attr in range(n_attr):
            attr_values = dataset[:, attr]

            if self.isFeatureCategorical(dataset, attr):
                split_datasets = self.split_categorical(dataset, attr)
                children = []
                for data in split_datasets:
                    if len(data) > 0:
                        children.append(data[:, -1])

                curr_info_gain = self.info_gain(dataset[:, -1], children)

                if curr_info_gain > max_info_gain:
                    best_split["feature_index"] = attr
                    # best_split["threshold"] = threshold
                    # best_split["datasets"] = data_left
                    # best_split["dataset_right"] = data_right
                    best_split["info_gain"] = curr_info_gain
                    max_info_gain = curr_info_gain

            else:
                possible_thresholds = np.unique(attr_values)
                for threshold in possible_thresholds:
                    data_left, data_right = self.split_continuous(dataset, attr, threshold)
                    if len(data_left) > 0 and len(data_right) > 0:
                        y, left_y, right_y = dataset[:, -1], data_left[:, -1], data_right[:, -1]
                        curr_info_gain = self.info_gain(y, [left_y, right_y])

                        if curr_info_gain > max_info_gain:
                            best_split["feature_index"] = attr
                            best_split["threshold"] = threshold
                            best_split["dataset_left"] = data_left
                            best_split["dataset_right"] = data_right
                            best_split["info_gain"] = curr_info_gain
                            max_info_gain = curr_info_gain

        return best_split

    def split_categorical(self, examples, idx):
        unique_categories = np.unique(examples[:, idx])
        datasets = {}
        for category in unique_categories:
            branch = np.array([row for row in examples if row[idx] == category])
            datasets[category] = branch

        return datasets

    def split_continuous(self, examples, idx, threshold):
        data_left = np.array([row for row in examples if row[idx] <= threshold])
        data_right = np.array([row for row in examples if row[idx] > threshold])
        return data_left, data_right

    def info_gain(self, parent, children):
        weighted_entropy_sum = 0
        for child in children:
            weight = len(child) / len(parent)
            entropy = self.entropy(child)
            weighted_entropy = weight * entropy
            weighted_entropy_sum += weighted_entropy

        gain = self.entropy(parent) - weighted_entropy_sum
        return gain

    def gain_ratio(self, gain, parent, children):
        intrinsic_value_sum = 0
        for child in children:
            len_ = len(child) / len(parent)
            intrinsic_value = len_ * np.log2(len_)
            intrinsic_value_sum += intrinsic_value

        g_ratio = gain / intrinsic_value_sum
        return g_ratio

    def gini_index(self, parent, children):
        gini_index = 0
        for child in children:
            len_ = len(child) / len(parent)
            gini = self.gini(child)
            gini_index += len_ * gini
        return gini_index

    def entropy(self, y):
        unique_labels = np.unique(y)
        entropy = 0
        for label in unique_labels:
            pk = len(y[y == label]) / len(y)
            entropy += -pk * np.log2(pk)
        return entropy

    def gini(self, y):
        unique_labels = np.unique(y)
        gini_sum = 0
        for label in unique_labels:
            pk = len(y[y == label]) / len(y)
            gini_sum += pk**2

        gini = 1 - gini_sum
        return gini
