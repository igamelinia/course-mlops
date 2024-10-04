# Packaging the ML Model of Classification

#### Problem Statement

- Company wants to automate the loan eligibility process based on customer detail provided while filling online application form.
- It is a classification problem where we have to predict whether a loan would be approved or not.

#### Data

The data corresponds to a set of financial requests associated with individuals.

The loan approval dataset is a collection of financial records and associated information used to determine the eligibility of individuals or organizations for obtaining loans from a lending institution. It includes various factors such as cibil score, income, employment status, loan term, loan amount, assets value, and loan status.

Source: Kaggle

## Directory structure

```bash
prediction_model


├── MANIFEST.in
├── prediction_model
│   ├── config
│   │   ├── config.py
│   │   └── __init__.py
│   ├── datasets
│   │   ├── __init__.py
│   │   ├── test.csv
│   │   └── train.csv
│   ├── __init__.py
│   ├── pipeline.py
│   ├── predict.py
│   ├── processing
│   │   ├── data_handling.py
│   │   ├── __init__.py
│   │   └── preprocessing.py
│   ├── trained_models
│   │   ├── classification.pkl
│   │   └── __init__.py
│   ├── training_pipeline.py
│   └── VERSION
├── README.md
├── requirements.txt
├── setup.py
└── tests
    ├── pytest.ini
    └── test_prediction.py
```

## Virtual Environment

Install virtualenv

```python
python3 -m pip install virtualenv
```

Check version

```python
virtualenv --version
```

Create virtual environment

```python
virtualenv ml_package
```

Activate virtual environment

For Linux/Mac

```python
source ml_package/bin/activate
```

For Windows

```python
ml_package\Scripts\activate
```

Deactivate virtual environment

```python
deactivate
```
