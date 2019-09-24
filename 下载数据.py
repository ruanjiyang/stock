import tushare as ts 
import numpy
ts.set_token('db42fb5372bce72ab61f22ef0a3310d5c441f09d17817f1cafd3ace2')


df = ts.pro_bar(ts_code='600036.SH', adj='qfq', start_date='20100101', end_date='20190923')
df.sort_values(by='trade_date', inplace=True)
df.to_csv('600036.csv',header=1) 
# df = ts.pro_bar(ts_code='399300.SZ', asset='I', start_date='20100101', end_date='20190923')
# print(df)