import pandas as pd
import numpy as np
import joblib
from pathlib import Path
import sys
import os

# Define Package root to avoid error modul not found
PACKAGE_ROOT = Path(os.path.abspath(os.path.dirname(__file__))).parent
sys.path.append(str(PACKAGE_ROOT))

from prediction_model.config import config
from prediction_model.processing.data_handling import load_pipeline, load_dataset, separate_data

classification_pipeline = load_pipeline(pipeline_to_load=config.MODEL_NAME)

def generate_predictions():
    test_data = load_dataset(file_name=config.TEST_FILE)
    X,y = separate_data(data=test_data)
    pred = classification_pipeline.predict(X)
    output = np.where(pred==1, 'Approved','Not Approved')
    print(output)
    return output

if __name__=='__main__':
    generate_predictions()

