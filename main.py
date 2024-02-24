import pandas as pd
from sklearn.metrics import f1_score, precision_score
from sklearn.model_selection import KFold

from decision_tree import DecisionTree, predict


def preprocess_data():
    column_names = ["A1", "A2", "A3", "A4", "A5", "A6", "A7", "A8", "A9", "A10", "A11", "A12", "A13", "A14", "A15", "A16"]
    df = pd.read_csv('training.data', header=None, names=column_names, delimiter=',')
    df_test = pd.read_csv('test.data', header=None, names=column_names, delimiter=',')
    cat_col = []
    cont_col = []

    columns_with_missing_values = df.columns[df.isin(['?']).any()].values
    for column in columns_with_missing_values:
        if (isinstance(df[column][0], (int, float)) or (df[column][0]).isdigit() or
                (df[column][0]).replace('.', '', 1).isnumeric()):
            cont_col.append(column)
        else:
            cat_col.append(column)

    for col in cat_col:
        median_val = df[col][df[col].size/2]
        df[col].replace('?', median_val, inplace=True)
        df_test[col].replace('?', median_val, inplace=True)
        df[col] = df[col].astype(object)
        df_test[col] = df_test[col].astype(object)

    for col in cont_col:
        df[col] = pd.to_numeric(df[col], errors='coerce')
        df_test[col] = pd.to_numeric(df_test[col], errors='coerce')
        if not pd.api.types.is_numeric_dtype(df[col]):
            print(f"Column '{col}' contains non-numeric values.")
        median_val = df[col].median(skipna=True)
        df[col] = df[col].fillna(median_val)
        df_test[col] = df_test[col].fillna(median_val)

    return df.columns.values, df, df_test


if __name__ == '__main__':
    columns, train_data, test_data = preprocess_data()
    num_folds = 10

    kf = KFold(n_splits=num_folds)

    best_f1 = 0
    best_model = None

    for train_index, val_index in kf.split(train_data):
        train_fold, val_fold = train_data.iloc[train_index], train_data.iloc[val_index]

        train_fold = train_fold.to_numpy()

        tree = DecisionTree(train_fold, method="cart").decision_tree_learning(train_fold, attributes=columns)

        predictions = []
        for _, example in val_fold.iterrows():
            predicted_class = predict(tree, example.values)
            predictions.append(predicted_class)
        y_true = val_fold.iloc[:, -1].values
        predictions = ['' if v is None else v for v in predictions]
        f1 = f1_score(y_true, predictions, average='weighted')
        print(f"F1: {f1}")
        if f1 > best_f1:
            best_f1 = f1
            best_model = tree
    best_model = DecisionTree(train_data).decision_tree_learning(train_data.values, attributes=columns)
    test_predictions = []
    for _, example in test_data.iterrows():
        predicted_class = predict(best_model, example.values)
        test_predictions.append(predicted_class)
    y_true_test = test_data.iloc[:, -1].values
    test_predictions = ['' if v is None else v for v in test_predictions]
    f1_test = f1_score(y_true_test, test_predictions, average='weighted')
    precision_score = precision_score(y_true_test, test_predictions, pos_label='+')

    print("Best Model F1 Score on Validation Set:", best_f1)
    print("F1 Score on Test Set:", f1_test)
    print("Precision score:", precision_score)
