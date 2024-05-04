# Heart Failure Prediction using AutoGluon

This repository contains code for predicting heart failure using AutoGluon, an automated machine learning (AutoML) library. AutoGluon automates the process of training and tuning machine learning models, making it easier to develop high-performing models with minimal manual effort.

# Results
![Results](https://github.com/lucaslov/heart-failure-prediction-autogluon/blob/main/result.png?raw=true)

## Requirements
- Python 3.x
- autogluon
- pandas
- scikit-learn

## Installation
You can install the required libraries using pip:

```bash
pip install -r requirements.txt
```

## Usage
1. Clone this repository:

```bash
git clone https://github.com/lucaslov/heart-failure-prediction-autogluon.git
cd heart-failure-prediction
```

2. Download the dataset `heart_failure_clinical_records_dataset.csv` and place it in the root directory of the repository.

3. Run the `ag.py` script:

```bash
python ag.py
```

## Description
- `ag.py`: This script loads the dataset, splits it into training and testing sets, and uses AutoGluon to train a model for predicting heart failure. The trained model is then used to make predictions on the test data, and the predictions along with the model leaderboard are printed.

## Dataset
The dataset used in this project is `heart_failure_clinical_records_dataset.csv`, containing various clinical features related to heart failure.

## References
- AutoGluon Documentation: https://auto.gluon.ai/stable/index.html
- Dataset Source: [Heart Failure Prediction Dataset](https://www.kaggle.com/datasets/andrewmvd/heart-failure-clinical-data)