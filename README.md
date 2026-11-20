# Password Strength Classifier

An exploratory machine-learning script that classifies password-strength labels using **character-level TF-IDF features and logistic regression**.

## Workflow

Read `data.csv`, drop missing examples, shuffle the rows, construct TF-IDF features, and train a multinomial logistic-regression classifier with an 80/20 split. The script prints a classification report, confusion matrix, and accuracy.

## Run

Use **Python 3.10** with the compatibility ranges in `requirements.txt`. These preserve the older pandas and scikit-learn APIs used by the script.

```sh
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python code.py
```

Run from the repository root. The CSV's first two columns must contain password text and strength labels, respectively. Training may use substantial memory for the supplied dataset. These historical dependency ranges have not been validated through a complete training run in the current environment.

## Evaluation limitations

TF-IDF is currently fitted before the train/test split, and the shuffle/split seeds are not fixed. Metrics are exploratory rather than a leakage-free, reproducible benchmark. A follow-up should split first, fit preprocessing on training examples through a scikit-learn pipeline, and use stratified deterministic evaluation.

The classifier is a dataset-based learning exercise; its output does not establish resistance to real password attacks.
