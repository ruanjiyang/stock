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
need_num = 100 #一般按周来算，选择5周.  按照天来算，选择60~90天。

stockCode="000001-weekly-index"
model=load_model(stockCode+".h5")

predict_days=3  #一共预测几天（不包括今天）

#训练数据的处理，我们选取整个数据集的前6000个数据作为训练数据，后面的数据为测试数据
#从csv读取数据
dataset = pd.read_csv(stockCode+'.csv')
dataset=dataset.fillna(0)
training_num=len(dataset)
features_num=dataset.shape[1]-3
dataset = dataset.iloc[:, 3:features_num+3].values
real_stock_price = dataset[training_num-20:]  #这是真实的数据。  -5表示取出五天前到今天的数据

sc = MinMaxScaler(feature_range=(0, 1))
sc.fit_transform(X=dataset[:training_num])  #这是为了和训练用的数据集保持一直的归一化。


dataset = pd.read_csv(stockCode+'.csv')
dataset=dataset.fillna(0)
dataset = dataset.iloc[training_num-need_num:training_num, 3:features_num+3].values

# print(dataset)
# print(dataset.shape)

for days in range(predict_days):   #填入延长预测的天数。（n
    #我们需要预测开盘数据，因此选取所有行、第三列数据
    #训练数据就是上面已经读取数据的前6000行
   # training_dataset = dataset[:training_num]
    #因为数据跨度几十年，随着时间增长，人民币金额也随之增长，因此需要对数据进行归一化处理
    #将所有数据归一化为0-1的范围

    '''
    fit_transform()对部分数据先拟合fit，
    找到该part的整体指标，如均值、方差、最大值最小值等等（根据具体转换的目的），
    然后对该trainData进行转换transform，从而实现数据的标准化、归一化等等。
    '''
    dataset_scaled = sc.transform(X=dataset)
    
    x_validation=np.zeros((days+1,need_num,features_num))
    for i in range(days+1):
        x_validation[i]=dataset_scaled[i:i+need_num]
    # print(x_validation)

    # print(x_validation.shape)


    predictes_stock_price = model.predict(x=x_validation)
    
    #使用 sc.inverse_transform()将归一化的数据转换回原始的数据，以便我们在图上进行查看
    predictes_stock_price = sc.inverse_transform(X=predictes_stock_price)
    dataset=np.append(dataset,predictes_stock_price[-1])
    print("新预测的到第%d天的总数据是=============="%(days+1),(predictes_stock_price))
    dataset=dataset.reshape((-1,features_num))



    
#绘制数据图表，红色是真实数据，蓝色是预测数据

#画在两张图上。
# plt.figure(12)
# plt.subplot(121)
# plt.plot(real_stock_price[:,0:1], color='red', label='Real Stock Close Price')

# plt.subplot(122)
# predictes_stock_price=np.vstack((real_stock_price[-1:,],predictes_stock_price))  #把真实的最后一天/周也添加在预测的输出中，作为右边的预测输出曲线的第一天。
# plt.plot(predictes_stock_price[:,0:1], color='blue', label='Predicted Stock Close Price(include today)',linestyle='--',marker='o')

# ### 画在一张图上。 
# predictes_stock_price=np.vstack((real_stock_price,predictes_stock_price))  #把真实的最后一天/周也添加在预测的输出中，作为右边的预测输出曲线的第一天。
# plt.plot(predictes_stock_price[:,0], color='blue', label='Predicted Stock Close Price',linestyle='--',marker='o')
# plt.plot(real_stock_price[:,0], color='red', label='Real Stock Close Price')

#画在六张图上。
predictes_stock_price=np.vstack((real_stock_price,predictes_stock_price))  #把真实的最后一天/周也添加在预测的输出中，作为右边的预测输出曲线的第一天。
plt.figure(23)
plt1=plt.subplot(231)
plt1.set_title('close',loc='right',fontstyle='italic')
plt.plot(predictes_stock_price[:,0], color='blue', label='1',linestyle='--',marker='.')
plt.plot(real_stock_price[:,0], color='red')

plt2=plt.subplot(232)
plt2.set_title('open',loc='right',fontstyle='italic')
plt.plot(predictes_stock_price[:,1], color='blue', label='2', linestyle='--',marker='.')
plt.plot(real_stock_price[:,1], color='red')

plt3=plt.subplot(233)
plt3.set_title('high',loc='right',fontstyle='italic')
plt.plot(predictes_stock_price[:,2], color='blue',label='3', linestyle='--',marker='.')
plt.plot(real_stock_price[:,2], color='red')

plt4=plt.subplot(234)
plt4.set_title('low',loc='right',fontstyle='italic')
plt.plot(predictes_stock_price[:,3], color='blue',label='4', linestyle='--',marker='.')
plt.plot(real_stock_price[:,3], color='red')

plt5=plt.subplot(235)
plt5.set_title('vol',loc='right',fontstyle='italic')
plt.plot(predictes_stock_price[:,4], color='blue',label='5', linestyle='--',marker='.')
plt.plot(real_stock_price[:,4], color='red')

plt6=plt.subplot(236)
plt6.set_title('amount',loc='right',fontstyle='italic')
plt.plot(predictes_stock_price[:,5], color='blue',label='6', linestyle='--',marker='.')
plt.plot(real_stock_price[:,5], color='red')

# plt.title(stockCode+'1-close,2-open,3-high,4-low,5-vol,6-amount')
# plt.xlabel(xlabel='Time')
# plt.ylabel(ylabel=stockCode+'  Stock Price')
plt.legend()
plt.show()