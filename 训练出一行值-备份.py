#https://blog.csdn.net/lpp5406813053/article/details/89788108

#数据预处理以及绘制图形需要的模块
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#构建长短时神经网络需要的方法
from sklearn.preprocessing import MinMaxScaler
from keras import callbacks,regularizers
from keras.models import Sequential,save_model,load_model
from keras.layers import Dense, LSTM, BatchNormalization,Activation,Dropout
from keras.optimizers import RMSprop,Adam
 

need_num = 12 #Time step，根据前need_num个数据来推测下一个数据。
epoch = 30
batch_size = 4  #batch_size越低， 预测精度越搞，曲线越曲折。
patience_times=5
stockCode="000001-weekly-index"
scale_rate=np.array([7000,7000,7000,7000,0.25*1e12,0.25*1e13])   #上证周K与月k线用。

#从csv读取数据
dataset = pd.read_csv(stockCode+'.csv')
dataset=dataset.fillna(0)
training_num=len(dataset)
features_num=dataset.shape[1]-3
dataset = dataset.iloc[:, 3:features_num+3].values  #从第4列开始读， 避开前三列。
training_dataset = dataset[:training_num]
training_dataset_scaled=training_dataset/scale_rate
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
model.add(LSTM(units=128,activation='softsign',dropout=0.001,recurrent_dropout=0.001, return_sequences=True,input_shape=[need_num, features_num]))   #activation='softsign',dropout=0.001,recurrent_dropout=0.001, stateful=False,return_sequences=True)(input)  #stateful=True,可以使得帧组之间产生关联。 记得要在fit时候，shuffle=True。
# model.add(BatchNormalization())

model.add(LSTM(units=64))
# model.add(BatchNormalization())
#kernel_regularizer=regularizers.l2(0.01),activity_regularizer=regularizers.l1(0.01)
model.add(Dense(64,kernel_regularizer=regularizers.l2(0.01),activity_regularizer=regularizers.l1(0.01)))
model.add(Activation('relu'))
model.add(Dropout(0.2))
model.add(Dense(32))
model.add(Activation('relu'))
model.add(Dropout(0.2))
model.add(Dense(24))
model.add(Activation('relu'))
model.add(Dropout(0.2))
model.add(Dense(12))
s_num))
#进行配置
adam=Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=True)
#AdaGrad
model.compile(optimizer=adam,loss='mean_squared_logarithmic_error')  #mean_absolute_error/mean_squared_error/mean_squared_logarithmic_error/

early_stopping=callbacks.EarlyStopping(monitor='val_loss',patience=patience_times, verbose=2, mode='auto')
model.fit(x=x_train, y=y_train,  epochs=epoch, batch_size=batch_size,validation_split=0.2,callbacks = [early_stopping])
model.save(stockCode+".h5")