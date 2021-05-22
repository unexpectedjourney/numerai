# Numerai
This project contains the solution for Numerai competiotion.
The __goal__: predict the stock market value based on abstract financial data.
The main __metrics__:
1) Spearman correlation;
2) Sharpe ration and its local modifications.

## Authors
- Arsenii Petryk
- Anton Babenko
- Volodymyr Prypeshniuk

## Configurations
To choose proper model or/and its hyperparams, please define them in `config/numeria.yml` file.

## Main pipeline functionalities:
### 1.Training
To train your model, please execute:
```
python main.py --train
```

### 2.Hyperparams tuning
To tune hyperparams of your model, please execute:
```
python main.py --tune_params
```
This command will call _hyperopt_ or _sklearn/RandomSearch_ estimators.

### 3.Prediction and scoring
To make prediction and score them according to our metrics, please execute:
```
python main.py --evaluate
```

### 4.Prediction for submission
To make prediction for the final submission, execute
```
python main.py --submit
```

## Project tree:
- assets - contains charts of experiments and era correlation results;
- config - contains different config file for proper work of the pipeline;
- data - contains the data files of the project;
- notebooks - contains notebooks of different experiments and examples;
- scripts - contains help scripts, e.g.: dataset resampling, etc;
- src - contains all code of our pipeline;
- submissions - contains submission files for numerai competition upload.
