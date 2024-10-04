import pytest
import numpy
from pathlib import Path
import os
import sys

#Note : u must set name file, function, class etc test with test_ in the front

# Adding the below path to avoid module not found error
PACKAGE_ROOT = Path(os.path.abspath(os.path.dirname(__file__))).parent
sys.path.append(str(PACKAGE_ROOT))

from prediction_model.processing.data_handling import load_pipeline, separate_data, load_dataset
from  prediction_model.config import config

#Fixtures --> functions before test function --> ensure single_prediction
classification_pipeline = load_pipeline(config.MODEL_NAME)

@pytest.fixture
def test_prediction():
    test_data = load_dataset(config.TEST_FILE)
    X,y = separate_data(test_data)
    pred = classification_pipeline.predict(X)
    return pred

def test_result_prediction_not_none(test_prediction):
    assert test_prediction is not None

def test_result_prediction_is_int(test_prediction):
    print(f'prediction[0]: {test_prediction[0]}, type: {type(test_prediction[0])}')
    assert isinstance(test_prediction[0], numpy.int64)