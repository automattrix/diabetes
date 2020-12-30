import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
import statsmodels.api as sm
from imblearn.over_sampling import SMOTE


def train_logistic_sklearn(df, features, target):
    print("Hello I am a ML model")

    X = df[features]
    y = df[target]

    oversample = SMOTE(random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )
    columns = X_train.columns
    oversample_X, oversample_y = oversample.fit_sample(X_train, y_train)
    oversample_X_df = pd.DataFrame(oversample_X, columns=columns)
    oversample_y_df = pd.DataFrame(oversample_y, columns=[target])

    # Create mode
    model = LogisticRegression(random_state=42)
    # Recursive feature selection
    rfe = RFE(model, 10)
    # Fit th emodel
    rfe.fit(X_train, y_train.values.ravel())
    print(rfe.support_)
    print(rfe.ranking_)

    # Model stats
    logit_model = sm.Logit(y, X)
    result = logit_model.fit()
    print(result.summary2())

    # Select columns from RFE
    mask = list(rfe.get_support())
    new_features = X.columns[mask]
    print(f"New features: {new_features}")

    # Fit model with new features
    X = df[new_features]
    # Model stats with new feautures
    logit_model = sm.Logit(y, X)
    result = logit_model.fit()
    print(result.summary2())

    print(type(new_features))
    zip_cols_mask = zip(X.columns, mask)

    # Only select column names where mask is True
    new_cols = [i for i, j in zip_cols_mask if j]

    new_cols.remove('delayed healing')
    new_features = X[new_cols].columns
    print(f"New features 2: {new_features}")

    # Fit model with new features minus >0.5 P
    X = df[new_features]
    # Model stats with new feautures
    logit_model = sm.Logit(y, X)
    result = logit_model.fit()
    print(result.summary2())


