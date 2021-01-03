from lib.data_processing import DataHolder


def main(csvpath):
    dia_data = DataHolder(csv_path=csvpath)
    dia_data.load_data()

    # Convert object cols to categorical cols
    cols_to_convert = dia_data.df.select_dtypes(include='object').columns.tolist()
    dia_data.convert_data(cols=cols_to_convert, mode='cat')

    # Convert age into age groups
    dia_data.create_bins(col_in='Age', col_out='age_bin')
    dia_data.convert_data(cols=['age_bin'], mode='cat')

    # Drop original age column
    cols_to_drop = ['Age']
    dia_data.drop_cols(cols=cols_to_drop)

    print(dia_data.df.head(15))

    # Train model
    target = 'class'
    features = dia_data.df.columns.tolist()
    features.remove(target)

    dia_data.train_model(mode='logistic_sklearn', features=features, target=target)


if __name__ == '__main__':
    csv_path = './data/01_raw/diabetes_data_upload.csv'
    main(csvpath=csv_path)
