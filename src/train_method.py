import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import GroupKFold, StratifiedKFold
from sklearn.preprocessing import KBinsDiscretizer

from src.preprocessing import preprocessing, drop_columns
from src.utils import (
    generate_model_name, unzip_dataframes,
    visualize_losses, zip_dataframes)
from src.split import TimeSeriesSplitGroups


def cross_validate(
    model,
    train_df,
    kfold,
    metric,
    target="target",
    model_name=None,
    test_df=None,
    preproc_funcs=None,
    viz_losses=None,
    copy_model=False,
    return_models=False,
    return_val_preds=False,
    *args,
    **kwargs
):
    if viz_losses is None:
        viz_losses = []
    if preproc_funcs is None:
        preproc_funcs = []
    if return_val_preds or viz_losses:
        val_preds = pd.DataFrame(
            {target: [0 for _ in range(train_df.shape[0])]})

    val_scores = []
    test_preds = []
    model_list = []
    loss_sample_dict = {}

    model_name = generate_model_name(model, model_name)

    if isinstance(kfold, GroupKFold):
        splits = kfold.split(train_df, groups=kwargs["groups"])
    elif isinstance(kfold, StratifiedKFold):
        target_values = train_df[[target]]
        est = KBinsDiscretizer(
            n_bins=50, encode='ordinal', strategy='quantile')
        stratify_on = est.fit_transform(target_values).T[0]
        splits = kfold.split(train_df, stratify_on)
    elif isinstance(kfold, TimeSeriesSplitGroups):
        eras = train_df.era
        splits = kfold.split(train_df, groups=eras)
    else:
        splits = kfold.split(train_df)

    train_df, test_df = drop_columns(train_df, test_df)

    for idx, (train_idx, val_idx) in enumerate(splits):
        tr_df = train_df.iloc[train_idx]
        val_df = train_df.iloc[val_idx]

        if preproc_funcs and test_df is not None:
            zip_df = zip_dataframes(val_df, test_df)
            (tr_df, zip_df) = preprocessing(tr_df, zip_df, preproc_funcs)
            (val_df, test_df) = unzip_dataframes(zip_df)
        if preproc_funcs and test_df is None:
            (tr_df, val_df) = preprocessing(tr_df, val_df, preproc_funcs)

        x_train = tr_df.drop(columns=target).values
        y_train = tr_df[target].values
        x_val = val_df.drop(columns=target).values
        y_val = val_df[target].values

        if copy_model:
            instance = clone(model)
        else:
            instance = model

        instance.fit(x_train, y_train)
        preds = instance.predict(x_val)
        model_list.append(instance)

        if return_val_preds or viz_losses:
            val_preds.iloc[
                val_df.index, val_preds.columns.get_loc(target)
            ] = preds

        fold_score = metric(y_val, preds)
        val_scores.append(fold_score)

        print(f"fold {idx+1} score: {fold_score}")

        if test_df is not None:
            if target in test_df.columns:
                test_df = test_df.drop(target, axis=1)
            test_fold_preds = instance.predict(test_df)
            test_preds.append(test_fold_preds)

    print(f"mean score: {np.mean(val_scores)}")
    print(f"score variance: {np.var(val_scores)}")

    if viz_losses:
        for loss in viz_losses:
            sample, initial_sample = loss(train_df[target], val_preds[target])
            loss_name = loss.__name__
            loss_sample_dict[loss_name] = (sample, initial_sample)

        visualize_losses(loss_sample_dict, model_name)

    if test_df is not None:
        return val_scores, test_preds

    if return_val_preds and return_models:
        return np.mean(val_scores), model_list, val_preds

    if return_val_preds:
        return np.mean(val_scores), val_preds

    if return_models:
        return np.mean(val_scores), model_list

    return np.mean(val_scores)

