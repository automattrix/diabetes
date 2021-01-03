import pandas as pd



def is_correct_prediction(prediction, observation):
    correct_prediction = None
    try:
        if prediction == observation:
            correct_prediction = True
        else:
            correct_prediction = False
    except Exception as e:
        print(e)
    return correct_prediction


def build_evaluation_df(prediction, probabilty, observed, features):
    """
    :param prediction: 1D array of model predictions
    :param probabilty: 2D array of prediction probability
    :param observed: 1D array of observed values
    :return:
    """
    df = None
    try:
        # print(features.head())
        # print(observed.head())
        features = features.copy()
        features['join_col'] = features.index
        df = pd.DataFrame(probabilty, columns=['prob_0', 'prob_1'])
        print(df.head())
        df['predictions'] = prediction
        df['observed'] = observed.tolist()
        df['correct_prediction'] = df[['predictions', 'observed']].apply(
            lambda x: is_correct_prediction(
                prediction=x['predictions'],
                observation=x['observed']),
            axis=1
        )
        df = df.copy()
        df['join_col'] = features.index
        df = df.merge(features, on=['join_col'])
        print(len(df.index))

    except Exception as e:
        print(e)

    return df
