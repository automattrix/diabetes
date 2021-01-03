import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

from evaluate.evaluate_model import build_evaluation_df
from graphing.log_model_performance import graph_log_performance, graph_gender_performance

from sklearn.feature_selection import RFE
import statsmodels.api as sm
from imblearn.over_sampling import SMOTE

from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, precision_score


def train_logistic_sklearn(df, features, target):

    X = df[features]
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    # First pass
    model = LogisticRegression(random_state=42)
    # Fit model
    model.fit(X_train, y_train)
    # Probability matrix of predicitons
    prob = model.predict_proba(X_test)
    # Model predictions
    prediction = model.predict(X_test)

    # Helper function to build dataframe used for graphing parts of model evaluation
    first_pass_evaluate_df = build_evaluation_df(prediction=prediction, probabilty=prob, observed=y_test, features=X_test)
    # Helpder function to generate graph to visualize model predictions
    graph_log_performance(df=first_pass_evaluate_df, name='first_pass_log')

    print(model.score(X_test, y_test))
    print(X_test.columns)
    print(model.coef_)
    print(prediction)
    print(first_pass_evaluate_df.head())

    print(f"Precision score: {precision_score(y_test, prediction)}")
    print(f"Accuracy score: {accuracy_score(y_test, prediction)}")
    print(f"Recall score: {recall_score(y_test, prediction)}")

    # Isolate incorrect cases
    incorrect_pred_df = first_pass_evaluate_df.loc[first_pass_evaluate_df['correct_prediction'] == False]
    print(incorrect_pred_df['Gender'].value_counts())


    correct_pred_df = first_pass_evaluate_df.loc[first_pass_evaluate_df['correct_prediction'] == True]

    cm = confusion_matrix(y_test, prediction)
    print(cm)

    print(incorrect_pred_df['Gender'].value_counts())
    print(correct_pred_df['Gender'].value_counts())

    graph_gender_performance(values=incorrect_pred_df['Gender'].value_counts(),
                             name='incorrect_predictions',
                             columns=['Gender', 'Count'])

    coefficients = list(model.coef_[0])
    inputs = X.columns
    coef_df = pd.DataFrame()
    coef_df['inputs'] = inputs
    coef_df['coef'] = coefficients
    coef_df = coef_df.sort_values(by='coef', ascending=False)
    print(coef_df)

    exit()

    # FUTURE REFINEMENTS...
    # SMOTE
    columns = X_train.columns
    oversample = SMOTE(random_state=42)
    oversample_X, oversample_y = oversample.fit_sample(X_train, y_train)
    oversample_X_df = pd.DataFrame(oversample_X, columns=columns)
    oversample_y_df = pd.DataFrame(oversample_y, columns=[target])

    # Create model
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


