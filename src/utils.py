import datetime
import os
import random

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px


def set_seed(seed):
    # torch.manual_seed(seed)
    # torch.cuda.manual_seed_all(seed)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)


def zip_dataframes(*dataframes):
    for idx, dataframe in enumerate(dataframes):
        dataframe["df_order"] = idx
    return pd.concat(dataframes)


def unzip_dataframes(dataframe):
    dataframes = []
    for n in dataframe["df_order"].unique().tolist():
        dataframes.append(
            dataframe[dataframe["df_order"] == n].drop(columns="df_order"))

    return dataframes


def visualize_losses(loss_sample_dict, model_name):
    sns.set()
    for name, sample_pair in loss_sample_dict.items():
        nomalized_sample = sample_pair[0]
        initial_sample = sample_pair[1]

        print(f"{name}: 10%={np.round(np.quantile(initial_sample, 0.1), 2)}"
              f" 90%={np.round(np.quantile(initial_sample, 0.9), 2)}")

        fig = px.histogram(x=nomalized_sample)
        fig.update_layout(title=f"{name}:"
                                f" 10%={np.round(np.quantile(initial_sample, 0.1), 2)},"
                                f" 90%={np.round(np.quantile(initial_sample, 0.9), 2)}")
        fig.write_image(f"losses/{model_name}_{name}.png")
        fig.show()


def plot_importance(feature_importances, features, num=20):
    feature_imp = pd.DataFrame(
        {'Value': feature_importances, 'Feature': features})
    plt.figure(figsize=(15, 7))
    sns.set(font_scale=1.4)
    sns.barplot(
        x="Value", y="Feature",
        data=feature_imp.sort_values(by="Value", ascending=False)[0:num])
    plt.title('Features (avg over folds)')
    plt.tight_layout()
    plt.savefig("importance.png")
    plt.show()


def generate_model_name(model, model_name=None):
    if model_name is None:
        model_name = {type(model).__name__}

    timestamp = int(datetime.datetime.now().timestamp())
    model_name = f"{model_name}_{timestamp}"
    return model_name


def clear_era(era):
    era = era.replace("era", "").strip()
    return int(era)
