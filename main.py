import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.metrics import f1_score
from sklearn.model_selection import KFold

from decision_tree import predict, DecisionTreeModel


def preprocess_data():
    column_names = ["A1", "A2", "A3", "A4", "A5", "A6", "A7", "A8", "A9", "A10", "A11", "A12", "A13", "A14", "A15",
                    "A16"]
    df = pd.read_csv("training.data", header=None, names=column_names, delimiter=",")
    df_test = pd.read_csv("test.data", header=None, names=column_names, delimiter=",")
    cat_col = []
    cont_col = []

    columns_with_missing_values = df.columns[df.isin(["?"]).any()].values
    for column in columns_with_missing_values:
        if (isinstance(df[column][0], (int, float)) or (df[column][0]).isdigit() or
                (df[column][0]).replace(".", "", 1).isnumeric()):
            cont_col.append(column)
        else:
            cat_col.append(column)

    for col in cat_col:
        median_val = df[col][df[col].size / 2]
        df[col] = df[col].replace("?", median_val)
        df_test[col] = df_test[col].replace("?", median_val)
        df[col] = df[col].astype(object)
        df_test[col] = df_test[col].astype(object)

    for col in cont_col:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        df_test[col] = pd.to_numeric(df_test[col], errors="coerce")
        median_val = df[col].median(skipna=True)
        df[col] = df[col].fillna(median_val)
        df_test[col] = df_test[col].fillna(median_val)

    return df.columns.values, df, df_test


def get_method():
    num = int(input("Enter 1 for CART or 2 for C4.5:"))
    if num == 1:
        val = "cart"
    elif num == 2:
        val = "c4.5"
    else:
        print("You entered an invalid number. Setting method as C4.5 by default")
        val = "c4.5"

    return val


def fit(data, method, columns):
    data = data.to_numpy()
    return DecisionTreeModel(data, method=method).decision_tree_learning(data, attributes=columns)


def decision_tree_kfold():
    columns, train_data, test_data = preprocess_data()
    method = get_method()
    kf = KFold(n_splits=10)

    best_f1_score = 0
    best_model = None

    scores = []

    for train_index, test_index in kf.split(train_data):
        train_fold, test_fold = train_data.iloc[train_index], train_data.iloc[test_index]
        tree = fit(train_fold, method, columns)

        predictions = []
        for _, example in test_fold.iterrows():
            predicted_class = predict(tree, example.values)
            predictions.append(predicted_class)
        y_true = test_fold.iloc[:, -1].values
        predictions = ["" if v is None else v for v in predictions]
        f1 = f1_score(y_true, predictions, average="weighted")
        scores.append(f1)
        if f1 > best_f1_score:
            best_f1_score = f1
            best_model = tree

    mean = np.mean(scores)

    test_predictions = []
    for _, example in test_data.iterrows():
        predicted_class = predict(best_model, example.values)
        test_predictions.append(predicted_class)
    y_true_test = test_data.iloc[:, -1].values
    test_predictions = ["" if v is None else v for v in test_predictions]
    f1_test = f1_score(y_true_test, test_predictions, average="weighted")

    plt.figure(figsize=(8, 5))
    plt.hist(scores, bins=10, color="skyblue", edgecolor="black")
    plt.xlabel("Accuracy Score")
    plt.ylabel("Frequency")
    plt.title(f"Distribution of Cross-Validation Accuracy Scores for {method.upper()}")
    plt.axvline(mean, color="red", linestyle="dashed", linewidth=2,
                label=f"Mean Accuracy: {mean:.2f}")
    plt.axvline(f1_test, color="orange", linestyle="dashed", linewidth=2,
                label=f"F1 test Accuracy: {f1_test:.2f}")
    plt.legend()
    plt.savefig("cross_val_scores.png")
    plt.show()

    print("Best Model F1 Score on KFold:", best_f1_score)
    print("F1 Score on Test Data:", f1_test)


if __name__ == "__main__":
    decision_tree_kfold()
