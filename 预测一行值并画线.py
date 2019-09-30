#https://blog.csdn.net/lpp5406813053/article/details/89788108

#数据预处理以及绘制图形需要的模块
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#构建长短时神经网络需要的方法
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential,save_model,load_model
from keras.layers import Dense, LSTM, BatchNormalization
 
#需要之前90次的数据来预测下一次的数据
need_num = 180 #一般按周来算，选择5周.  按照天来算，选择60~90天。
epoch = 20
batch_size = 4  #batch_size越低， 预测精度越搞，曲线越曲折。
patience_times=6
stockCode="600036.SH"

#scale_rate=np.array([7000,7000,7000,7000,0.25*1e12,0.25*1e13])   #上证周K线用。
#scale_rate=np.array([3000,3000,3000,3000,1e8,1e9]) #上证月K线用。


model=load_model(stockCode+".h5")

predict_days=50  #一共预测几天（不包括今天）

dataset = pd.read_csv(stockCode+'.csv')
dataset=dataset.fillna(0)
training_num=len(dataset)
features_num=dataset.shape[1]-3
dataset = dataset.iloc[:, 3:features_num+3].values
real_stock_price = dataset  #这是真实的数据。  -5表示取出五天前到今天的数据


dataset = pd.read_csv(stockCode+'.csv')
dataset=dataset.fillna(0)
dataset = dataset.iloc[:, 3:features_num+3].values

sc = MinMaxScaler(feature_range=(0, 1))
dataset_scaled = sc.fit_transform(X=dataset[:training_num])

for days in range(predict_days):   #填入延长预测的天数。（n
    #dataset_scaled=dataset/scale_rate  #使用我定义的 scale_rate
    dataset_scaled = sc.transform(X=dataset)
    x_validation=[]
    for i in range(need_num, dataset_scaled.shape[0]+1):
        x_validation.append(dataset_scaled[i-need_num: i])
    x_validation = np.array(x_validation)
    x_validation = np.reshape(x_validation, (-1, need_num, features_num))
    predictes_stock_price = model.predict(x=x_validation)
    
    #predictes_stock_price=predictes_stock_price*scale_rate #使用我的scale_rate
    predictes_stock_price = sc.inverse_transform(X=predictes_stock_price)
    dataset=np.vstack((dataset,predictes_stock_price[-1]))
    


   
predictes_stock_price=np.vstack((real_stock_price[0:need_num],predictes_stock_price))

print(predictes_stock_price[-1])

print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
print(real_stock_price[-1])

plt.figure(11)  #23
total_days_on_grid=400
plt.xticks(np.arange(0,total_days_on_grid+1,1))
plt.grid(axis='x',linestyle='-.')
plt1=plt.subplot(111) #231
plt1.set_title('close',loc='right',fontstyle='italic')
plt.plot(predictes_stock_price[-total_days_on_grid-predict_days:-1,0], color='blue', label='1',linestyle='-')
plt.plot(real_stock_price[-total_days_on_grid:-1,0], color='red')
plt.legend()
plt.show()


## 检查下判断涨跌的准确率
successful_rate=0
predictes_raise=False
real_stock_price_raise=False
for i in range(need_num,training_num-1):
    if predictes_stock_price[i+1][0]-predictes_stock_price[i][0] > 0: #涨
        predictes_raise=True
    else:
        predictes_raise=False
    
    if real_stock_price[i+1][0]-real_stock_price[i][0] > 0: #涨
        real_stock_price_raise=True
    else:
        real_stock_price_raise=False
    
    if predictes_raise==real_stock_price_raise:
        successful_rate=successful_rate+1
    
print("总的涨跌预测正确率:",successful_rate/(training_num-2-need_num)*100, "%" )