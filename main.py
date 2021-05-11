from pathlib import Path
from pprint import pprint

import fire
import yaml

from src.trainer import Trainer
from src.utils import set_seed

import warnings

warnings.filterwarnings("ignore")

from joblib import load


def load_model_from_file(path):
    model = load(path)

    return model


def load_config(config_path):
    with open(config_path) as file:
        config = yaml.full_load(file)
    pprint(f"Config: {config}")
    return config


def main(
        train=False,
        tune_params=False,
        evaluate=False,
        submit=False,
        model_name=None,
        save=False,
        test=False,
):
    config_path = Path("configs") / "numerai.yml"
    config = load_config(config_path)

    seed = config.get("seed")
    model_params = config.get("model_params")

    if model_name is None:
        model_name = config.get("model_name")

    if tune_params:
        print("Wiping model params")
        model_params = None

    set_seed(seed)
    dir_path = Path("./data")
    train_file = dir_path / "train.csv"
    test_file = dir_path / "test.csv"
    save_path = Path("./models")
    submissions_path = Path("./submissions")

    trainer = Trainer(
        train_file,
        test_file,
        submissions_path,
        model_name,
        model_params,
        save_path
    )
    if train:
        trainer.train()
    elif tune_params:
        trainer.find_hyperparameters()
    elif evaluate:
        trainer.evaluate()
    elif submit:
        trainer.evaluate_for_submition()
    else:
        print("Bye!")
    if save:
        trainer.save_model()
    if test:
        print('loading...')
        trainer.load_model('models/xgboost_9_5_2021.pickle')
        print('loaded!')
        trainer.evaluate()


if __name__ == '__main__':
    fire.Fire(main)
