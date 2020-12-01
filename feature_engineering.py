import pandas as pd
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

def melt_train_data(train_data):
    train_data = pd.melt(train_data, id_vars = ['id', 'item_id', 'dept_id', 'cat_id', 'store_id', 'state_id'], var_name = 'day', value_name = 'demand')
    train_data = reduce_mem_usage(train_data)
    
    return train_data

def load_train_data():
    calendar, sell_prices, train_data = read_data()

    train_data = melt_train_data(train_data)
    print(train_data.head())




load_train_data()