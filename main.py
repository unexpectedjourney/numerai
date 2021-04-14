from pathlib import Path
from pprint import pprint

import fire
import yaml

from src.trainer import Trainer
from src.utils import set_seed


def load_config(config_path):
    with open(config_path) as file:
        config = yaml.full_load(file)
    pprint(f"Config: {config}")
    return config


def main(
        train=False,
        tune_params=False,
        evaluate=False,
):
    config_path = Path("configs") / "numerai.yml"
    config = load_config(config_path)

    seed = config.get("seed")
    model_name = config.get("model_name")
    model_params = config.get("model_params")

    set_seed(seed)
    dir_path = Path("./data")
    train_file = dir_path / "numerai_training_data.csv"
    test_file = dir_path / "numerai_tournament_data.csv"
    submissions_path = Path("./submissions")

    trainer = Trainer(
        train_file,
        test_file,
        submissions_path,
        model_name,
        model_params
    )
    if train:
        trainer.train()
    elif tune_params:
        trainer.find_hyperparameters()
    elif evaluate:
        trainer.evaluate()
    else:
        print("Bye!")


if __name__ == '__main__':
    fire.Fire(main)
