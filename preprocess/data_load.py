import pandas as pd


def load_csv(csv_path):
    df = None
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(e)

    return df


def convert_to_categories(df, col):
    df[col] = df[col]
    try:
        df[col] = pd.Categorical(df[col])
        df[col] = df[col].cat.codes
    except Exception as e:
        print(f"Unable to convert column {col} to cat codes...:{e}")

    return df[col]


def preprocess_csv(csv_path):
    df = load_csv(csv_path=csv_path)
    cols_to_cat = df.columns.tolist()
    cols_to_remove = ['Age']
    for element in cols_to_remove:
        cols_to_cat.remove(element)


def convert_to_bins(array_in, start, end, step):
    bins = None
    bin_range = range(start, end, step)
    label_range = range(start, (end - step), step)
    try:
        labels = [f"{x+1}s" for x in label_range]
        print(labels)
        bins = pd.cut(x=array_in, bins=bin_range, labels=labels)
    except Exception as e:
        print(e)

    return bins
