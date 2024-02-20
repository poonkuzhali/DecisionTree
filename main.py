import pandas as pd


def preprocess_data():
    columns = ["A1", "A2", "A3", "A4", "A5", "A6", "A7", "A8", "A9", "A10", "A11", "A12", "A13", "A14", "A15", "A16"]

    df = pd.read_csv('training.data', header=None, names=columns, delimiter=',')
    df_test = pd.read_csv('test.data', header=None, names=columns, delimiter=',')
    cat_col = ["A1", "A4", "A5", "A6", "A7", "A14"]
    cont_col = ["A2"]

    for col in cat_col:
        max_val = df[col].sort_values().max()
        df[col].replace('?', max_val, inplace=True)
        df_test[col].replace('?', max_val, inplace=True)
        df[col] = df[col].astype(object)
        df_test[col] = df_test[col].astype(object)

    for col in cont_col:
        df[col].replace("?", pd.NA, inplace=True)
        df[col].replace('?', pd.NA, inplace=True)
        median_val = df[col].median
        df[col] = df[col].fillna(median_val)
        df[col] = pd.to_numeric(df[col], errors='coerce')
        df_test[col] = df_test[col].fillna(median_val)
        df_test[col] = pd.to_numeric(df_test[col], errors='coerce')

    return df, df_test


if __name__ == '__main__':
    train_data, test_data = preprocess_data()
