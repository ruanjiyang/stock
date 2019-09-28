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
need_num = 30
#训练数据的大小
training_num = 0
#迭代10次
epoch = 40
batch_size = 8  #batch_size越低， 预测精度越搞，曲线越曲折。
features_num=9
patience_times=5
stockCode="000001-index"



#训练数据的处理，我们选取整个数据集的前6000个数据作为训练数据，后面的数据为测试数据
#从csv读取数据
dataset = pd.read_csv(stockCode+'.csv')
dataset=dataset.fillna(0)
training_num=len(dataset)
dataset = dataset.iloc[:, 3:features_num+3].values




#我们需要预测开盘数据，因此选取所有行、第三列数据
#训练数据就是上面已经读取数据的前6000行
# training_dataset = dataset[:training_num]
#因为数据跨度几十年，随着时间增长，人民币金额也随之增长，因此需要对数据进行归一化处理
#将所有数据归一化为0-1的范围
sc = MinMaxScaler(feature_range=(0, 1))
sc.fit_transform(X=dataset)
'''
fit_transform()对部分数据先拟合fit，
找到该part的整体指标，如均值、方差、最大值最小值等等（根据具体转换的目的），
然后对该trainData进行转换transform，从而实现数据的标准化、归一化等等。
'''
#training_dataset_scaled = sc.fit_transform(X=training_dataset)


model=load_model(stockCode+".h5")

#进行测试数据的处理
#前6000个为测试数据，但是将5910，即6000-90个数据作为输入数据，因为这样可以获取
#测试数据的潜在规律
inputs = dataset[-need_num:len(dataset),:]
print(inputs.shape)


#这里使用的是transform而不是fit_transform，因为我们已经在训练数据找到了
#数据的内在规律，因此，仅使用transform来进行转化即可
inputs = sc.transform(X=inputs)  #归一化处理
inputs = inputs.reshape(1,need_num, features_num)
print(inputs)

#这是真实的股票价格，是源数据的[6000:]即剩下的231个数据的价格

#进行预测
predictes_stock_price = model.predict(x=inputs)


print("新预测的数据是==============",(predictes_stock_price))




