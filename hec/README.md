# Data
This project is based on the data from
https://www.kaggle.com/robikscube/hourly-energy-consumption/home
Download the data, unzip and put the csv files in ./data/ to run notebook.

# Setup
Run setup.py to setup
$ python setup.py install

Test will be given by following command.
$ python setup.py test

# Files
- README.md
    - This file.
- notebook
    - Analysis.ipynb
        - notebook file to analyse the provided data.
    - Prediction.ipynb
        - notebook file to demonstrate the power prediction.
    - plotter.py
        - Contains visualisation functions.
- hec/utils
    - data_handler.py
        - Contains utility functions and a class to handle data.
    - eval.py
        - Evaluation tool
    - logger.py
        - logging tool
- hec/model
    - model.py
        - model modules to learn and predict
    - preprocessing.py
        - preprocessing for training and prediction
- hec/apps
    - main_train.py
        - script to train the model
    - main_test_prediction.py
        - script to precdict on test data
    - predict_api.py
        - RESTful API to provide prediction


# how to run main scripts
## main_train.py
`$ python main_train.py DATA_CSV_FILE_NAME MODE_PRM_NAME NORM_PRM_NAME`

## main_test_predict.py
`$ python main_test_predict.py DATA_CSV_FILE_NAME MODEL_PRM_NAME NORM_PRM_NAME`

## predict_api.py
`$ python predict_api.py MODEL_PRM_NAME NORM_PRM_NAME`