import pandas as pd
from os.path import abspath, dirname, join

data_dir = join(abspath(dirname(dirname(__file__))), "data")
train_df_path = join(data_dir, "numerai_training_data.csv")
test_df_path = join(data_dir, "numerai_tournament_data.csv")

final_train_path = join(data_dir, "train.csv")
final_test_path = join(data_dir, "test.csv")


def main():
    print("=" * 30, "\nReading")
    train_df = pd.read_csv(train_df_path)
    testing_df = pd.read_csv(test_df_path)
    validation_df = testing_df[testing_df.data_type == "validation"]
    test_df = testing_df

    print("=" * 30, "\nAppending")
    train_df = train_df.append(validation_df)

    # era part
    print("=" * 30, "\nEra sorting")
    train_df["era_number"] = train_df["era"].apply(lambda x: x[3:])
    train_df["era_number"] = train_df["era_number"].astype(int)
    train_df = train_df.sort_values("era_number")
    train_df = train_df.drop("era_number", axis=1)

    # save part
    print("=" * 30, "\nSaving")
    train_df.to_csv(final_train_path, index=False)
    test_df.to_csv(final_test_path, index=False)


if __name__ == "__main__":
    main()
