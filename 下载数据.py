import tushare as ts 

# import numpy as np 
# import sys
# import numpy as np
# import keras
# from keras.models import Sequential, Model,load_model
# from keras.layers import Dense, Dropout,Activation,Input
# from keras.layers import Embedding, LSTM,SimpleRNN, Reshape
# from keras.optimizers import RMSprop,Adam
# from keras.utils import np_utils
# from keras.callbacks import TensorBoard,EarlyStopping


hs300DataHist=(ts.get_hist_data('399300'))
hs300DataHist.sort_values(by='date', inplace=True)  #按照日期，从早到晚排序

hs300DataHist.to_csv('Result.csv',header=1) 
