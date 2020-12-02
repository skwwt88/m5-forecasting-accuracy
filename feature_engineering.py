import pandas as pd
import gc
from utils import reduce_mem_usage

def read_data():
    print('Reading files...')
    calendar = pd.read_csv('./m5-forecasting-accuracy/calendar.csv')
    calendar = reduce_mem_usage(calendar)
    print('Calendar has {} rows and {} columns'.format(calendar.shape[0], calendar.shape[1]))
    sell_prices = pd.read_csv('./m5-forecasting-accuracy/sell_prices.csv')
    sell_prices = reduce_mem_usage(sell_prices)
    print('Sell prices has {} rows and {} columns'.format(sell_prices.shape[0], sell_prices.shape[1]))
    train_data = pd.read_csv('./m5-forecasting-accuracy/sales_train_validation.csv')
    
    print('Sales train validation has {} rows and {} columns'.format(train_data.shape[0], train_data.shape[1]))

    return calendar, sell_prices, train_data

def melt_train_data(input_data):
    data = pd.melt(input_data, id_vars = ['id', 'item_id', 'dept_id', 'cat_id', 'store_id', 'state_id'], var_name = 'day', value_name = 'demand')
    data = reduce_mem_usage(data)

    print('melt_train_data has {} rows and {} columns'.format(data.shape[0], data.shape[1]))
    del input_data
    gc.collect()

    return data

def merge_with_calendar(input_data, calendar):
    data = pd.merge(input_data, calendar, how = 'left', left_on = ['day'], right_on = ['d'])
    data.drop(['d', 'day'], inplace = True, axis = 1)
    print('merge_with_calendar has {} rows and {} columns'.format(data.shape[0], data.shape[1]))

    del input_data
    gc.collect()

    return data

def merge_with_sales(input_data, sell_prices):
    data = input_data.merge(sell_prices, on = ['store_id', 'item_id', 'wm_yr_wk'], how = 'left')
    print('merge_with_sales has {} rows and {} columns'.format(data.shape[0], data.shape[1]))
    
    del input_data
    gc.collect()

    return data

def transform_data(data):
    
    # rolling demand features
    
    
    # price features
    data['lag_price_t1'] = data.groupby(['id'])['sell_price'].transform(lambda x: x.shift(1))
    data['lag_t28'] = data.groupby(['id'])['demand'].transform(lambda x: x.shift(28))
    data['rolling_price_std_t7'] = data.groupby(['id'])['sell_price'].transform(lambda x: x.rolling(7).std())
    data['rolling_price_std_t30'] = data.groupby(['id'])['sell_price'].transform(lambda x: x.rolling(30).std())
    
    # time features
    data['date'] = pd.to_datetime(data['date'])
    data['month'] = data['date'].dt.month
    data['week'] = data['date'].dt.week
    data['day'] = data['date'].dt.day
    data['dayofweek'] = data['date'].dt.dayofweek
    
    return data

def load_train_data():
    calendar, sell_prices, train_data = read_data()
    train_data = melt_train_data(train_data)
    train_data = merge_with_calendar(train_data, calendar)
    train_data = merge_with_sales(train_data, sell_prices)
    transform_data(train_data)

    print(train_data.head())

    return train_data