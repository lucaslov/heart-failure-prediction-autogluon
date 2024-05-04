from autogluon.tabular import TabularDataset, TabularPredictor
import pandas as pd
from sklearn.model_selection import train_test_split

data = pd.read_csv('heart_failure_clinical_records_dataset.csv')
train, test = train_test_split(data)

train_data = TabularDataset(train)
test_data = TabularDataset(test)

predictor = TabularPredictor(label='DEATH_EVENT', path='models', eval_metric='roc_auc')\
    .fit(train_data, presets='medium_quality', time_limit=60)
    
predictions=predictor.predict(test_data)
print(predictions)
print(predictor.leaderboard())