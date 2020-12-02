from feature_engineering import load_train_data
from dask_ml import preprocessing
import lightgbm as lgb
import pandas as pd
import gc
from sklearn.metrics import mean_squared_error
from utils import reduce_mem_usage

features = ['item_id', 'month', 'dayofweek', 
            'snap_CA', 'snap_TX', 'snap_WI', 'sell_price', 'lag_price_t1', 'lag_t28', 'rolling_price_std_t7', 'rolling_price_std_t30']

cat_features = ['item_id',  'month', 'dayofweek', 
            'snap_CA', 'snap_TX', 'snap_WI']

features_1 = ['item_id',  'month', 'dayofweek', 
            'snap_CA', 'snap_TX', 'snap_WI', 'sell_price', 'lag_price_t1', 'lag_t28', 'rolling_price_std_t7', 'rolling_price_std_t30', 'date', 'demand']

def transform(data):
    for feature in cat_features:
        encoder = preprocessing.LabelEncoder()
        data[feature] = encoder.fit_transform(data[feature])
    return encoder,data

# %%
def run_lgb(data):
    
    # going to evaluate with the last 28 days
    x_train = data[data['date'] <= '2015-03-27']
    y_train = x_train['demand']
    x_val = data[data['date'] > '2015-03-27']
    y_val = x_val['demand']

    del data
    gc.collect()

    # define random hyperparammeters
    params = {
        'boosting_type': 'gbdt',
        'metric': 'rmse',
        'objective': 'regression',
        'n_jobs': -1,
        'seed': 236,
        'learning_rate': 0.1,
        'bagging_fraction': 0.75,
        'bagging_freq': 10, 
        'colsample_bytree': 0.75}

    train_set = lgb.Dataset(x_train[features], y_train, categorical_feature=cat_features)
    val_set = lgb.Dataset(x_val[features], y_val, categorical_feature=cat_features)
    
    del x_train, y_train

    model = lgb.train(params, train_set, num_boost_round = 2500, early_stopping_rounds = 50, valid_sets = [train_set, val_set], verbose_eval = 100)
    return model

data = load_train_data()
data.to_csv('train.csv')

input = pd.read_csv('./train.csv', usecols=features_1)
data = reduce_mem_usage(input)
_, data = transform(data)

del input
gc.collect()

run_lgb(data)