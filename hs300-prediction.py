import tushare as ts 

import numpy as np 
import sys
import socket 
from PyQt5.QtWidgets import QApplication , QMainWindow
#class Ui_MainWindow(QtWidgets.QMainWindow):  #用这个替换Ui_Mind_locker_Ui.py 的 class Ui_MainWindow(object):
import numpy as np
import random
from PyQt5.QtWidgets import QFileDialog
from PyQt5.QtCore import QFileInfo
import keras
from keras.models import Sequential, Model,load_model
from keras.layers import Dense, Dropout,Activation,Input
from keras.layers import Embedding, LSTM,SimpleRNN, Reshape
from keras.optimizers import RMSprop,Adam
from keras.utils import np_utils
from keras.callbacks import TensorBoard,EarlyStopping
import matplotlib.pyplot as plt


hs300DataHist=(ts.get_hist_data('399300'))
hs300DataHist=np.array(hs300DataHist) #转为numpy
totalHistDay=601
time_steps=60 
Feature_number=1
hs300DataHist=hs300DataHist[0:totalHistDay]   # 取最近的600天数据。
print("一共有多少天的历史记录：",totalHistDay)
newHS300DataHist=np.zeros((totalHistDay,13)) # 数据的倒序， 最早的数据放在最开头，这样训练的时候可以有序进行。

for i in range(totalHistDay):
    # 数据的倒序， 最早的数据放在最开头，这样训练的时候可以有序进行。
    newHS300DataHist[i]=hs300DataHist[totalHistDay-i-1] 
newHS300DataHist=newHS300DataHist[:,3]
x=newHS300DataHist[-time_steps-1:-1].reshape([1,time_steps,Feature_number])
#x=hs300DataHist[:time_steps+1].reshape([1,time_steps,Feature_number])
print(x.shape)
print(x)

model = load_model('hs300.h5')
preds = model.predict(x)
print(preds)
# ########################开始搭建神经网络#############################
# model=Sequential()
# training_batch_size=20
# model.add(LSTM(100,activation='softsign',input_shape = (20, Feature_number),dropout=0.2,recurrent_dropout=0.1, stateful=False,return_sequences=False))  #stateful=True,可以使得帧组之间产生关联。 记得要在fit时候，shuffle=True。
# # model.add(LSTM(200,activation='tanh',input_shape = (time_steps, total_EEG_Features),dropout=0.2,recurrent_dropout=0.1, stateful=False,return_sequences=False))  #stateful=True,可以使得帧组之间产生关联。 记得要在fit时候，shuffle=True。

# # model.add(Dense(100,kernel_regularizer=regularizers.l2(0.002),bias_regularizer=regularizers.l2(0.002)))
# # model.add(Activation('relu'))
# # model.add(Dropout(0.2))

# # model.add(Dense(100,kernel_regularizer=regularizers.l2(0.002),bias_regularizer=regularizers.l2(0.002)))
# # model.add(Activation('relu'))
# # model.add(Dropout(0.2))

# # model.add(Dense(100,kernel_regularizer=regularizers.l2(0.002),bias_regularizer=regularizers.l2(0.002)))
# # model.add(Activation('relu'))
# # model.add(Dropout(0.2))

# model.add(Dense(100))
# model.add(Activation('relu'))
# model.add(Dropout(0.2))
# model.add(Dense(100))
# model.add(Activation('relu'))
# model.add(Dropout(0.2))
# model.add(Dense(100))
# model.add(Activation('relu'))
# model.add(Dropout(0.2))
# #model.add(Dropout(0.2))

# model.add(Dense(Feature_number))
# model.add(Activation('linear'))
# #rmsprop=RMSprop(lr=0.001, rho=0.9, epsilon=1e-6)
# adam=Adam(lr=0.02, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
# # model.compile(loss='binary_crossentropy',optimizer=adam,metrics=['accuracy'])
# model.compile(loss='mean_squared_error',optimizer='adam',metrics=['accuracy'])

# ########################神经网络搭建完毕#############################


# ################这个tb，是为了使用TensorBoard########################
# tb = TensorBoard(log_dir='./logs',  # log 目录
#     histogram_freq=0,  # 按照何等频率（epoch）来计算直方图，0为不计算. 好像只能设置为0，否则程序死机。
#     batch_size=32,     # 用多大量的数据计算直方图
#     write_graph=True,  # 是否存储网络结构图
#     write_grads=False, # 是否可视化梯度直方图
#     write_images=False,# 是否可视化参数
#     embeddings_freq=0, 
#     embeddings_layer_names=None, 
#     embeddings_metadata=None)    

# # 在命令行，先conda activate envs，然后进入本代码所在的目录，然后用 tensorboard --logdir=logs/ 来看log
# # 然后打开chrome浏览器，输入http://localhost:6006/ 来查看
# # 如果出现tensorboard错误，那么需要修改 ...\lib\site-packages\tensorboard\manager.py，其中keras环境下的这个文件，我已经修改好了。
# ########################开始训练#############################
# #training_loop_times=100  # 把进度条分为10分，所以训练也分解为 10次。
# # for i in range(training_loop_times):  #这个for，只是为了进度条的显示，所以分成 10次来训练。
# #     per_step_result=model.fit(x, y,validation_split=0.33, epochs=int(max(training_times/training_loop_times,1)), batch_size=training_batch_size,verbose = 1,shuffle=True) #这一行没有带callbacks，所以无法使用TensorBoard
# #     final_result_loss=str(per_step_result.history['loss'][0])[:5]
# #     final_result_acc=str(per_step_result.history['acc'][0])[:5]
# #     print("Training loop times:",i,"/100")
# #     ui.label.setText("开始机器学习你的脑纹。目前的损失率为"+final_result_loss+"  目前的准确率为"+final_result_acc)
# #     ui.progressBar.setProperty("value", (i+1)*100/training_loop_times)
# #     QApplication.processEvents()  #用于PyQt界面的刷新，保证流畅程度。

# training_times=1000
# training_batch_size=5
# early_stopping = EarlyStopping(monitor='val_loss',patience=int(training_times*0.2),verbose=1,mode='auto') 
# # training_process_bar=Training_process_bar()
# #per_training_step_result=model.fit(x, y, validation_split=0.33,epochs=training_times, batch_size=training_batch_size,verbose = 1,shuffle=True,callbacks=[training_process_bar]) #这一行带callbacks，是为了使用TensorBoard
# per_training_step_result=model.fit(x, y, validation_split=0.33,epochs=training_times, batch_size=training_batch_size,verbose = 1,shuffle=False) #这一行带callbacks，是为了使用TensorBoard

# model.save('hs300.h5')


# # directly_load_model_flag= False

