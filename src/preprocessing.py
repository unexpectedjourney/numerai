from src.utils import clear_era


def preprocessing(train_df, test_df, funcs):
    train_df = train_df.copy()
    test_df = test_df.copy()
    for func in funcs:
        train_df, test_df = func(train_df, test_df)
    return train_df, test_df



def get_validation_data(train_df, test_df):
    test_df = test_df[test_df.data_type == "validation"]
    return train_df, test_df


def drop_columns(train_df, test_df=None):
    columns = ['id', 'data_type', 'era']
    train_df = train_df.drop(columns=columns, axis=1)
    if test_df:
        test_df = test_df.drop(columns=columns, axis=1)
    return train_df, test_df


def clear_era_records(train_df, test_df):
    train_df.era = train_df.era.apply(clear_era)
    train_df = train_df.sort_values("era")
    return train_df, test_df
