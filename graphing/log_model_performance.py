import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

GRAPH_DIR = './data/08_reporting/'


def graph_log_performance(df, name):
    sns.set(style='whitegrid')
    fig, ax = plt.subplots(figsize=(10, 5))
    df = df.sort_values(by='prob_1', ascending=True)
    df = df.reset_index()
    df['index_col'] = df.index
    # sns.scatterplot(data=df, x='index_col', y='prob_1')
    sns.lineplot(data=df, x='index_col', y='prob_1', ax=ax)
    sns.scatterplot(data=df, x='index_col', y='predictions', hue='correct_prediction',
                    palette={True: 'deepskyblue', False: 'tomato'}, ax=ax)
    ax.axhline(0.5, ls='--')
    ax.set_xlabel("Predictions", fontsize=14)
    ax.set_ylabel("Predicted Probability", fontsize=14)
    text_x = (len(df.index) * .8)
    ax.text(text_x, 0.75, "0: Negative Diagnosis\n 1: Positive Diagnosis", fontsize=11)

    sns.despine()

    save_path = f"{GRAPH_DIR}{name}.png"
    plt.savefig(save_path)


def graph_gender_performance(values, name, columns):
    sns.set(style='whitegrid')
    fig, ax = plt.subplots(figsize=(5, 5))

    remap_dict = {0: 'Female', 1: 'Male'}
    df = pd.DataFrame(values)
    df = df.reset_index()
    df.columns = columns
    df['Gender'] =  df['Gender'].map(remap_dict)
    print(type(values))
    print(df)

    sns.barplot(data=df, x='Gender', y='Count', palette=['tomato', 'tomato'])

    ax.set_xlabel("Gender", fontsize=14)
    ax.set_ylabel("Count", fontsize=14)
    text_x = (len(df.index) * .8)

    # ax.text(text_x, 0.75, "0: Negative Diagnosis\n 1: Positive Diagnosis", fontsize=11)

    sns.despine()

    save_path = f"{GRAPH_DIR}{name}.png"
    plt.savefig(save_path)



