import tushare as ts 
import numpy
ts.set_token('db42fb5372bce72ab61f22ef0a3310d5c441f09d17817f1cafd3ace2')
pro = ts.pro_api('db42fb5372bce72ab61f22ef0a3310d5c441f09d17817f1cafd3ace2')

stockCode='601258.SH'
set_start_date='20010101'
set_end_date='20191231'

# ### 下载前除权的个股数据
# df = ts.pro_bar(ts_code=stockCode, adj='qfq', start_date=set_start_date, end_date=set_end_date)
# df.sort_values(by='trade_date', inplace=True)
# df.to_csv(stockCode+'.csv',header=1) 

# ## 下载个股资金流向
# pro = ts.pro_api('db42fb5372bce72ab61f22ef0a3310d5c441f09d17817f1cafd3ace2')
# df = pro.moneyflow(ts_code=stockCode, start_date=set_start_date, end_date=set_end_date)
# df.sort_values(by='trade_date', inplace=True)
# df.to_csv(stockCode+'-mf.csv',header=1) 

### 下载指数数据
# pro = ts.pro_api('db42fb5372bce72ab61f22ef0a3310d5c441f09d17817f1cafd3ace2')
# df = pro.index_daily(ts_code='399300.SZ', start_date=set_start_date, end_date=set_end_date)
# df.sort_values(by='trade_date', inplace=True)
# df.to_csv('399300-index.csv',header=1) 

# pro = ts.pro_api('db42fb5372bce72ab61f22ef0a3310d5c441f09d17817f1cafd3ace2')
# df = pro.index_daily(ts_code='399001.SZ', start_date=set_start_date, end_date=set_end_date)
# df.sort_values(by='trade_date', inplace=True)
# df.to_csv('399001-index.csv',header=1) 

# pro = ts.pro_api('db42fb5372bce72ab61f22ef0a3310d5c441f09d17817f1cafd3ace2')
# df = pro.index_daily(ts_code='000001.SH', start_date=set_start_date, end_date=set_end_date)
# df.sort_values(by='trade_date', inplace=True)
# df.to_csv('000001-index.csv',header=1) 


#指数周k
pro = ts.pro_api('db42fb5372bce72ab61f22ef0a3310d5c441f09d17817f1cafd3ace2')
df = pro.index_weekly(ts_code='000001.SH',adj='qfq', start_date=set_start_date, end_date=set_end_date, fields='ts_code,trade_date,open,high,low,close,vol,amount')
df.sort_values(by='trade_date', inplace=True)
df.to_csv('000001-weekly-index.csv',header=1) 


#指数月k
pro = ts.pro_api('db42fb5372bce72ab61f22ef0a3310d5c441f09d17817f1cafd3ace2')
df = pro.index_monthly(ts_code='000001.SH',adj='qfq',start_date=set_start_date, end_date=set_end_date,  fields='ts_code,trade_date,open,high,low,close,vol,amount')
df.sort_values(by='trade_date', inplace=True)
df.to_csv('000001-monthly-index.csv',header=1) 