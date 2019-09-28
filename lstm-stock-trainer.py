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
from keras.optimizers import RMSprop,Adam
 
#需要之前90次的数据来预测下一次的数据
need_num = 90
#训练数据的大小
training_num = 0
#迭代10次
epoch = 30
batch_size = 8  #batch_size越低， 预测精度越搞，曲线越曲折。
features_num=27
patience_times=5
stockCode="601258.SH"

#训练数据的处理，我们选取整个数据集的前6000个数据作为训练数据，后面的数据为测试数据
#从csv读取数据
dataset = pd.read_csv(stockCode+'.csv')
dataset=dataset.fillna(0)
training_num=len(dataset)-1
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
model.add(BatchNormalization())

model.add(LSTM(units=50))
model.add(BatchNormalization())

model.add(Dense(30))
model.add(Activation('relu'))
model.add(Dropout(0.1))
model.add(Dense(30))
model.add(Activation('relu'))
model.add(Dropout(0.1))

model.add(Dense(units=features_num))

#进行配置
adam=Adam(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=True)
#adam=Adam(lr=0.005, beta_1=0.9, beta_2=0.999, epsilon=None, amsgrad=True)
model.compile(optimizer=adam,loss='mean_squared_error')

print(x_train.shape,y_train.shape)
early_stopping=callbacks.EarlyStopping(monitor='val_loss',patience=patience_times, verbose=2, mode='min')
#min_delta=0,baseline=None, restore_best_weights=False)

model.fit(x=x_train, y=y_train,  epochs=epoch, batch_size=batch_size,validation_split=0.2,callbacks = [early_stopping])
model.save(stockCode+".h5")
