#https://blog.csdn.net/lpp5406813053/article/details/89788108

#数据预处理以及绘制图形需要的模块
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#构建长短时神经网络需要的方法
from sklearn.preprocessing import MinMaxScaler
from keras import callbacks
from keras.models import Sequential,save_model,load_model
from keras.layers import Dense, LSTM, BatchNormalization,Activation,Dropout
 
#需要之前90次的数据来预测下一次的数据
need_num = 90
#训练数据的大小
training_num = 2350
#迭代10次
epoch = 20
batch_size = 4  #batch_size越低， 预测精度越搞，曲线越曲折。
features_num=9
 
#训练数据的处理，我们选取整个数据集的前6000个数据作为训练数据，后面的数据为测试数据
#从csv读取数据
dataset = pd.read_csv('600519.csv')
dataset = dataset.iloc[:, 3:features_num+3].values  #从第4列开始读， 避开前三列。

#我们需要预测开盘数据，因此选取所有行、第三列数据
#训练数据就是上面已经读取数据的前6000行
training_dataset = dataset[:training_num]
#因为数据跨度几十年，随着时间增长，人民币金额也随之增长，因此需要对数据进行归一化处理
#将所有数据归一化为0-1的范围
sc = MinMaxScaler(feature_range=(0, 1))
'''
fit_transform()对部分数据先拟合fit，
找到该part的整体指标，如均值、方差、最大值最小值等等（根据具体转换的目的），
然后对该trainData进行转换transform，从而实现数据的标准化、归一化等等。
'''
training_dataset_scaled = sc.fit_transform(X=training_dataset)

x_train = []
y_train = []
#每90个数据为一组，作为测试数据，下一个数据为标签
for i in range(need_num, training_dataset_scaled.shape[0]):
    x_train.append(training_dataset_scaled[i-need_num: i])
    y_train.append(training_dataset_scaled[i, :])
#将数据转化为数组
x_train, y_train = np.array(x_train), np.array(y_train)
#因为LSTM要求输入的数据格式为三维的，[training_number, time_steps, 1]，因此对数据进行相应转化
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], features_num))

#构建网络，使用的是序贯模型
model = Sequential()
#return_sequences=True返回的是全部输出，LSTM做第一层时，需要指定输入shape
model.add(LSTM(units=50,return_sequences=True,input_shape=[x_train.shape[1], features_num]))
# model.add(BatchNormalization())

model.add(LSTM(units=50))
#model.add(BatchNormalization())

model.add(Dense(30))
model.add(Activation('relu'))
#model.add(Dropout(0.2))
model.add(Dense(30))
model.add(Activation('relu'))
#model.add(Dropout(0.2))


model.add(Dense(units=features_num))
#进行配置
model.compile(optimizer='adam',
            loss='mean_squared_error')

print(x_train.shape,y_train.shape)
early_stopping=callbacks.EarlyStopping(monitor='val_loss',patience=3, verbose=2, mode='auto')
#min_delta=0,baseline=None, restore_best_weights=False)

model.fit(x=x_train, y=y_train,  epochs=epoch, batch_size=batch_size,validation_split=0.2,callbacks = [early_stopping])
model.save("lstm-stock-example.h5")
model=load_model("lstm-stock-example.h5")

#进行测试数据的处理
#前6000个为测试数据，但是将5910，即6000-90个数据作为输入数据，因为这样可以获取
#测试数据的潜在规律
inputs = dataset[training_num - need_num:]

inputs = inputs.reshape(-1, features_num)
#这里使用的是transform而不是fit_transform，因为我们已经在训练数据找到了
#数据的内在规律，因此，仅使用transform来进行转化即可
inputs = sc.transform(X=inputs)
x_validation = []

for i in range(need_num, inputs.shape[0]):
    x_validation.append(inputs[i - need_num:i, :])

x_validation = np.array(x_validation)
x_validation = np.reshape(x_validation, (x_validation.shape[0], x_validation.shape[1], features_num))

#这是真实的股票价格，是源数据的[6000:]即剩下的231个数据的价格
real_stock_price = dataset[training_num:]
#进行预测
predictes_stock_price = model.predict(x=x_validation)
#使用 sc.inverse_transform()将归一化的数据转换回原始的数据，以便我们在图上进行查看
predictes_stock_price = sc.inverse_transform(X=predictes_stock_price)


    
#绘制数据图表，红色是真实数据，蓝色是预测数据
plt.plot(real_stock_price[:,3:4], color='red', label='Real Stock Close Price')
#print('Real Stock Close Price',real_stock_price[:,2:3])
plt.plot(predictes_stock_price[:,3:4], color='blue', label='Predicted Stock Close Price')

# plt.plot(real_stock_price[:,0:1], color='darkred', label='Real Stock Open Price')
# #print('Real Stock Open Price',real_stock_price[:,0:1])
# plt.plot(predictes_stock_price[:,0:1], color='darkblue', label='Predicted Stock Open Price')


plt.title(label='ShangHai Stock Price Prediction')
plt.xlabel(xlabel='Time')
plt.ylabel(ylabel='ShangHai Stock Price')
plt.legend()
plt.show()