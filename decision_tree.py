import numpy
import numpy as np

from importance_formulas import gain_ratio_categorical, gain_ratio_continuous, gini_index_categorical, gini_index_continuous, \
    gain_continuous

COLUMNS = ["A1", "A2", "A3", "A4", "A5", "A6", "A7", "A8", "A9", "A10", "A11", "A12", "A13", "A14", "A15", "A16"]


def remove_all(item, seq):
    if isinstance(seq, str):
        return seq.replace(item, '')
    elif isinstance(seq, set):
        rest = seq.copy()
        rest.remove(item)
        return rest
    else:
        return [x for x in seq if x != item]


def predict(tree, example):
    if isinstance(tree, Branches):
        attr_val = example[tree.attr]
        if attr_val in tree.branches:
            return predict(tree.branches[attr_val], example)
        else:
            return tree.default_child(example)
    elif isinstance(tree, Leaf):
        return tree.result


class Leaf:
    def __init__(self, result):
        self.result = result

    def __call__(self, example):
        return self.result


def classify_data(data):
    label_column = data[:, -1]
    unique_classes, counts_unique_classes = np.unique(label_column, return_counts=True)

    index = counts_unique_classes.argmax()
    classification = unique_classes[index]

    return classification


class Branches:
    def __init__(self, attr, attr_name=None, default_child=None, branches=None):
        self.attr = attr
        self.attr_name = attr_name or attr
        self.default_child = default_child
        self.branches = branches or {}

    def add(self, val, subtree):
        self.branches[val] = subtree


class DecisionTree:
    def __init__(self, dataset, branches=None, max_depth=5, min_samples=2, method="c4.5"):
        self.branches = branches or {}
        self.max_depth = max_depth
        self.min_samples = min_samples
        self.dataset = dataset
        self.method = method

    def isFeatureCategorical(self, dataset, idx):
        data_type = type(dataset[:, idx][0])
        if data_type == np.object_ or data_type == str:
            return True
        else:
            return False

    def plurality_value(self, examples):
        y = np.array(examples)
        max_count = 0
        popular = None
        unique_labels, label_counts = np.unique(y[:, -1], return_counts=True)

        for label, count in zip(unique_labels, label_counts):
            print(f"Label: {label}, Count: {count}")
            if count > max_count:
                max_count = count
                popular = label

        return Leaf(popular)

    def all_same_class(self, examples):
        y = np.array(examples)
        label = y[0, -1]

        for e in y:
            if e[-1] != label:
                return False

        return True

    def decision_tree_learning(self, dataset, attributes=None, parent_examples=None, depth=0):
        if len(dataset) == 0:
            return self.plurality_value(parent_examples)
        if self.all_same_class(dataset):
            return Leaf(dataset[0, -1])
        if len(attributes) == 0:
            return self.plurality_value(dataset)
        if (len(dataset) < self.min_samples) or (depth == self.max_depth):
            classification = classify_data(dataset)
            return classification
        else:
            attr_idx = -1
            if self.method == "c4.5":
                max_info_gain = -float("inf")
                for i in range(len(attributes) - 1):
                    if self.isFeatureCategorical(dataset, i):
                        gain_ratio = gain_ratio_categorical(dataset, i)
                    else:
                        gain_ratio = gain_ratio_continuous(dataset, i)

                    if gain_ratio > max_info_gain:
                        max_info_gain = gain_ratio
                        attr_idx = i
            elif self.method == "cart":
                min_gini = 9999999
                for i in range(len(attributes) - 1):
                    if self.isFeatureCategorical(dataset, i):
                        gini_index = gini_index_categorical(dataset, i)
                    else:
                        gini_index, t = gini_index_continuous(dataset, i)

                    if gini_index < min_gini:
                        min_gini = gini_index
                        attr_idx = i

            tree = Branches(attr_idx, COLUMNS[attr_idx], self.plurality_value(dataset))
            depth += 1
            if self.isFeatureCategorical(dataset, attr_idx):
                for (v_k, exs) in self.split_function_cat(attr_idx, dataset):
                    exs = numpy.asarray(exs)
                    subtree = self.decision_tree_learning(exs, remove_all(attributes[attr_idx], attributes), dataset,
                                                          depth)
                    tree.add(v_k, subtree)
            else:
                for (v_k, exs) in self.split_function_cont(attr_idx, dataset):
                    exs = numpy.asarray(exs)
                    subtree = self.decision_tree_learning(exs, remove_all(attributes[attr_idx], attributes), dataset,
                                                          depth)
                    tree.add(v_k, subtree)
            return tree

    def split_function_cat(self, attr_idx, dataset):
        spl = []
        column = dataset[:, attr_idx]
        unique_values = np.unique(column)

        for v in unique_values:
            examples = []
            for data in dataset:
                if data[attr_idx] == v:
                    examples.append(data)
            spl.append((v, examples))

        return spl

    def split_function_cont(self, attr_idx, dataset):
        best_threshold = None

        if self.method == "c4.5":
            info_gain, best_threshold = gain_continuous(dataset, attr_idx)
        elif self.method == "cart":
            gini, best_threshold = gini_index_continuous(dataset, attr_idx)

        splits = [(f"<= {best_threshold}", dataset[dataset[:, attr_idx] <= best_threshold]),
                  (f"> {best_threshold}", dataset[dataset[:, attr_idx] > best_threshold])]
        return splits
