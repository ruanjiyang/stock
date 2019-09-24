import tushare as ts 
import numpy
ts.set_token('db42fb5372bce72ab61f22ef0a3310d5c441f09d17817f1cafd3ace2')

### 下载前除权的个股数据
# df = ts.pro_bar(ts_code='600519.SH', adj='qfq', start_date='20100101', end_date='20190924')
# df.sort_values(by='trade_date', inplace=True)
# df.to_csv('600519.csv',header=1) 

### 下载指数数据
df = ts.pro_bar(ts_code='399300.SZ', asset='I', start_date='20100101', end_date='20190924')
df.sort_values(by='trade_date', inplace=True)
df.to_csv('399300.csv',header=1) 
print(df)