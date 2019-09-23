import tushare as ts 

import numpy as np 
import sys
import numpy as np
import keras
from keras.models import Sequential, Model,load_model
from keras.layers import Dense, Dropout,Activation,Input
from keras.layers import Embedding, LSTM,SimpleRNN, Reshape
from keras.optimizers import RMSprop,Adam
from keras.utils import np_utils
from keras.callbacks import TensorBoard,EarlyStopping


hs300DataHist=(ts.get_hist_data('399300'))
hs300DataHist=np.array(hs300DataHist) #转为numpy
hs300DataHist=hs300DataHist[:,3]
Feature_number=1
totalHistDay=601 #一共采样多少天。
time_steps=60  #根据多少天的连续数据来预测下一天的数据
hs300DataHist=hs300DataHist[0:totalHistDay]   # 取最近的600天数据。
print("一共有多少天的历史记录：",totalHistDay)
newHS300DataHist=np.zeros((totalHistDay,Feature_number)) # 数据的倒序， 最早的数据放在最开头，这样训练的时候可以有序进行。
for i in range(totalHistDay):
    # 数据的倒序， 最早的数据放在最开头，这样训练的时候可以有序进行。
    newHS300DataHist[i]=hs300DataHist[totalHistDay-i-1] 


# ##为了测试，构造一个 很简单的数据集合
# for i in range(totalHistDay):
#     # 数据的倒序， 最早的数据放在最开头，这样训练的时候可以有序进行。
#     for j in range(Feature_number):
#         newHS300DataHist[i][j]=i 

# print (newHS300DataHist)

x=np.zeros([totalHistDay-time_steps,time_steps,Feature_number])  #开始构造训练集x，30组，每组time_steps天，Feature_number个特征值）。按照每天来滑动x的窗口。
for i in range(totalHistDay-time_steps):
    for j in range(time_steps):
        x[i][j]=newHS300DataHist[i+j-1+1]

print("x的形状:",x.shape)
print(x)

y=np.zeros([totalHistDay-time_steps,Feature_number])
for i in range(totalHistDay-time_steps):
    for j in range(Feature_number):
        y[i][j]=newHS300DataHist[i+time_steps][j]
print("y的形状:",y.shape)
print(y)

# y=np.zeros([600-time_steps+1,time_steps,Feature_number])

# y=np.zeros([600-time_steps+1,1])


########################开始搭建神经网络#############################
model=Sequential()
training_batch_size=5
model.add(LSTM(100,activation='tanh',input_shape = (time_steps, Feature_number),dropout=0.2,recurrent_dropout=0.1, stateful=False,return_sequences=False))  #stateful=True,可以使得帧组之间产生关联。 记得要在fit时候，shuffle=False.

# model.add(Dropout(0.2))

model.add(Dense(100))
model.add(Activation('relu'))
model.add(Dropout(0.2))
model.add(Dense(100))
model.add(Activation('relu'))
model.add(Dropout(0.2))
model.add(Dense(100))
model.add(Activation('relu'))
model.add(Dropout(0.2))
model.add(Dense(100))
model.add(Activation('relu'))
model.add(Dropout(0.2))
model.add(Dense(100))
model.add(Activation('relu'))
model.add(Dropout(0.2))
#model.add(Dropout(0.2))

model.add(Dense(Feature_number))
model.add(Activation('linear'))
#rmsprop=RMSprop(lr=0.001, rho=0.9, epsilon=1e-6)
#adam=Adam(lr=0.01, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
# model.compile(loss='binary_crossentropy',optimizer=adam,metrics=['accuracy'])
model.compile(loss='mean_squared_error',optimizer='adam',metrics=['mae'])

########################神经网络搭建完毕#############################


################这个tb，是为了使用TensorBoard########################
tb = TensorBoard(log_dir='./logs',  # log 目录
    histogram_freq=0,  # 按照何等频率（epoch）来计算直方图，0为不计算. 好像只能设置为0，否则程序死机。
    batch_size=32,     # 用多大量的数据计算直方图
    write_graph=True,  # 是否存储网络结构图
    write_grads=False, # 是否可视化梯度直方图
    write_images=False,# 是否可视化参数
    embeddings_freq=0, 
    embeddings_layer_names=None, 
    embeddings_metadata=None)    


training_times=100
training_batch_size=580
#early_stopping = EarlyStopping(monitor='val_loss',patience=int(training_times*0.2),verbose=1,mode='auto') 
per_training_step_result=model.fit(x, y, validation_split=0.33,epochs=training_times, batch_size=training_batch_size,verbose = 1,shuffle=True) #这一行带callbacks，是为了使用TensorBoard
#per_training_step_result=model.fit(x, y, epochs=training_times, batch_size=training_batch_size,verbose = 1,shuffle=False) #这一行带callbacks，是为了使用TensorBoard

model.save('hs300.h5')

