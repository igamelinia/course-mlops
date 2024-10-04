import pathlib
import os
import sys

# # Path ke direktori proyek
# project_path = os.path.abspath('D:\Meli\Repositori local\course-mlops\Packaging-ML-Model\packaging-ml-model')

# # Menambahkan path proyek ke sys.path
# sys.path.append(project_path)

# import prediction_model

# PACKAGE_ROOT = pathlib.Path(prediction_model.__file__).resolve().parent

PACKAGE_ROOT = pathlib.Path(os.path.abspath(os.path.dirname(__file__))).resolve().parent.parent
sys.path.append(str(PACKAGE_ROOT))

import prediction_model

# make sure path must under prediction_model direktori
DATAPATH = os.path.join(PACKAGE_ROOT, 'prediction_model', 'datasets')
# DATAPATH = os.path.join(PACKAGE_ROOT,"datasets")

FILE_NAME = 'loan_approval_dataset.csv'
TEST_FILE = 'test_data.csv'

MODEL_NAME = 'classification_model.pkl'
SAVE_MODEL_PATH = os.path.join(PACKAGE_ROOT,'prediction_model','trained_models')

TARGET = 'loan_status'

# Final features used in the model
FEATURES = ['no_of_dependents', 'education', 'self_employed', 'income_annum',
       'loan_amount', 'loan_term', 'cibil_score', 'residential_assets_value',
       'commercial_assets_value', 'luxury_assets_value', 'bank_asset_value', 'loan_status']

PRED_FEATURES = ['no_of_dependents', 'education', 'self_employed', 'income_annum',
       'loan_amount', 'loan_term', 'cibil_score', 'residential_assets_value',
       'commercial_assets_value', 'luxury_assets_value', 'bank_asset_value']

NUM_FEATURES = ['ApplicantIncome', 'LoanAmount', 'Loan_Amount_Term']

CAT_FEATURES = ['education', 'self_employed']

# Dict for transformation
FEATURES_TO_ENCODE = {
    'education' : ['Graduate'],
    'self_employed' : ['Yes']
}

NEW_FEATURE_ADD = 'total_assets_value'
FEATURE_TO_ADD = ['residential_assets_value','commercial_assets_value', 'luxury_assets_value','bank_asset_value' ]

DROP_FEATURES = ['residential_assets_value','commercial_assets_value', 'luxury_assets_value','bank_asset_value' ]

# taking log of numerical columns
LOG_FEATURES = ['income_annum','loan_amount','total_assets_value'] 




