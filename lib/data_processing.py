from preprocess.data_load import load_csv, convert_to_categories, convert_to_bins
from train.train_logistic import train_logistic_sklearn
import pandas as pd


class DataHolder:
    def __init__(self, csv_path):
        self.csv_path = csv_path
        self.df = None
        self.model = None

    def dropna(self):
        df = self.df.dropna()
        return df

    def drop_cols(self, cols):
        for col in cols:
            try:
                self.df = self.df.drop([col], axis=1)
            except Exception as e:
                print(e)

    def load_data(self):
        self.df = load_csv(self.csv_path)

    def convert_data(self, cols, mode):
        if mode == 'cat':
            for col in cols:
                self.df[col] = convert_to_categories(df=self.df, col=col)
        elif mode == 'obj':
            print("not implemented yet")
        else:
            raise Exception("Select valid mode: ['cat']")

    def create_bins(self, col_in, col_out):
        self.df[col_out] = convert_to_bins(array_in=self.df[col_in], start=9, end=120, step=10)

    def train_model(self, mode, features, target):
        if mode == 'logistic_sklearn':
            self.model = train_logistic_sklearn(df=self.df, features=features, target=target)
