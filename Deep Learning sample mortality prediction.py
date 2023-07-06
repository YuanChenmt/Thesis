#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
df=pd.read_csv('Div1_fltper_5.csv')
print(df)


# In[2]:


# data preprocessing
data=df.filter(['Year','Age','mx'])

data1=data[:1368]
data2=data1[168:]

mortality=data2.filter(['mx'])

np.set_printoptions(suppress=True)
mortality1=np.array(mortality).reshape(50,24)


# In[ ]:





# In[3]:





# In[4]:


# normalize
from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
sca_x=scaler.fit_transform(mortality1)
print(sca_x)


# In[5]:


pre_datas=10
mem_history_datas=3
from collections import deque
deq=deque(maxlen=mem_history_datas)

x=[]
for i in sca_x:
    deq.append(list(i))
    if len(deq)==mem_history_datas:
        x.append(list(deq))

x_last=x[-pre_datas:]
x=x[:-pre_datas]
print(len(x))
print(len(x_last))
        
y=sca_x[mem_history_datas-1:-pre_datas]


# In[6]:


x=np.array(x)
y=np.array(y)
print(x.shape)
print(y.shape)
x_train=x[:30]
x_test=x[30:]
y_train=y[:30]
y_test=y[30:]


# In[7]:


import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional, GRU

def lstm(units):
    model = Sequential()
    model.add(LSTM(units,activation='tanh',input_shape=(x.shape[1],x.shape[2]),return_sequences=True))


    model.add((LSTM(units,activation='tanh',return_sequences=True)))
    model.add(Dropout(0.2))

    model.add((LSTM(units,activation='tanh',return_sequences=False)))
    model.add(Dropout(0.2))
    
    model.add(Dense(y_train.shape[1]))

    model.compile(optimizer='adam',
                 loss='mse',
                 metrics=['mape'])
    return model



def bilstm(units):
    model = Sequential()
    model.add(Bidirectional(LSTM(units,activation='tanh',return_sequences=True),input_shape=(x.shape[1],x.shape[2])))


    model.add(Bidirectional(LSTM(units,activation='tanh',return_sequences=True)))
    model.add(Dropout(0.2))

    model.add(Bidirectional(LSTM(units,activation='tanh',return_sequences=False)))
    model.add(Dropout(0.2))

    model.add(Dense(y_train.shape[1]))

    model.compile(optimizer='adam',
                 loss='mse',
                 metrics=['mape'])
    return model

def bigru(units):
    model = Sequential()
    model.add(Bidirectional(GRU(units,activation='tanh',return_sequences=True),input_shape=(x.shape[1],x.shape[2])))


    model.add(Bidirectional(GRU(units,activation='tanh',return_sequences=True)))
    model.add(Dropout(0.2))

    model.add(Bidirectional(GRU(units,activation='tanh',return_sequences=False)))
    model.add(Dropout(0.2))

    model.add(Dense(y_train.shape[1]))

    model.compile(optimizer='adam',
                 loss='mse',
                 metrics=['mape'])
    return model


def gru(units):
    model = Sequential()
    model.add(GRU(units,activation='tanh',return_sequences=True,input_shape=(x.shape[1],x.shape[2])))


    model.add(GRU(units,activation='tanh',return_sequences=True))
    model.add(Dropout(0.2))

    model.add(GRU(units,activation='tanh',return_sequences=False))
    model.add(Dropout(0.2))

    model.add(Dense(y_train.shape[1]))

    model.compile(optimizer='adam',
                 loss='mse',
                 metrics=['mape'])
    return model

model_lstm=lstm(128)
model_bilstm=bilstm(128)
model_gru=gru(128)
model_bigru=bigru(128)


# In[8]:


model_lstm.summary()


# In[9]:


model_bigru.summary()


# In[10]:


model_bilstm.summary()


# In[11]:


model_gru.summary()


# In[12]:


def fit_model(model):
    history= model.fit(x_train,y_train,epochs=50,batch_size=32,verbose=1)
    return history

history_lstm=fit_model(model_lstm)
history_bilstm=fit_model(model_bilstm)
history_gru = fit_model(model_gru)
history_bigru = fit_model(model_bigru)


# In[13]:


y_test = scaler.inverse_transform(y_test)
y_train = scaler.inverse_transform(y_train)

def prediction(model):
    prediction = model.predict(x_test)
    prediction = scaler.inverse_transform(prediction)
    return prediction

prediction_lstm = prediction(model_lstm)

def prediction(model):
    prediction = model.predict(x_test)
    prediction = scaler.inverse_transform(prediction)
    return prediction

prediction_bilstm = prediction(model_bilstm)

def prediction(model):
    prediction = model.predict(x_test)
    prediction = scaler.inverse_transform(prediction)
    return prediction

prediction_gru = prediction(model_gru)

def prediction(model):
    prediction = model.predict(x_test)
    prediction = scaler.inverse_transform(prediction)
    return prediction

prediction_bigru = prediction(model_bigru)


# In[14]:


np.set_printoptions(suppress=True)
print(prediction_lstm)


# In[15]:


np.set_printoptions(suppress=True)
print(prediction_bigru)


# In[16]:


print(prediction_bilstm)


# In[17]:


print(prediction_gru)


# In[18]:


def evaluate_prediction(predictions, actual, model_name):
    errors = predictions - actual
    mse = np.square(errors).mean()
    rmse = np.sqrt(mse)
    mae = np.abs(errors).mean()
    print(model_name + ':')
    print('Mean Absolute Error: {:.8f}'.format(mae))
    print('Root Mean Square Error: {:.8f}'.format(rmse))
    print('')
evaluate_prediction(prediction_lstm, y_test, 'LSTM')

def evaluate_prediction(predictions, actual, model_name):
    errors = predictions - actual
    mse = np.square(errors).mean()
    rmse = np.sqrt(mse)
    mae = np.abs(errors).mean()
    print(model_name + ':')
    print('Mean Absolute Error: {:.8f}'.format(mae))
    print('Root Mean Square Error: {:.8f}'.format(rmse))
    print('')
evaluate_prediction(prediction_bilstm, y_test, 'Bidirectiona LSTM')

def evaluate_prediction(predictions, actual, model_name):
    errors = predictions - actual
    mse = np.square(errors).mean()
    rmse = np.sqrt(mse)
    mae = np.abs(errors).mean()
    print(model_name + ':')
    print('Mean Absolute Error: {:.8f}'.format(mae))
    print('Root Mean Square Error: {:.8f}'.format(rmse))
    print('')
evaluate_prediction(prediction_bigru, y_test, 'Bidirectiona GRU')

def evaluate_prediction(predictions, actual, model_name):
    errors = predictions - actual
    mse = np.square(errors).mean()
    rmse = np.sqrt(mse)
    mae = np.abs(errors).mean()
    print(model_name + ':')
    print('Mean Absolute Error: {:.8f}'.format(mae))
    print('Root Mean Square Error: {:.8f}'.format(rmse))
    print('')
evaluate_prediction(prediction_gru, y_test, 'GRU')


# In[19]:


def evaluate_prediction(predictions, actual, model_name):
    errors = predictions - actual
    me = (errors).mean()
    print(model_name + ':')
    print('Mean Error: {:.8f}'.format(me))
    print('')
evaluate_prediction(prediction_gru, y_test, 'GRU')
evaluate_prediction(prediction_bilstm, y_test, 'Bidirectiona LSTM')
evaluate_prediction(prediction_lstm, y_test, 'LSTM')
evaluate_prediction(prediction_bigru, y_test, 'Bidirectiona GRU')


# In[20]:


def evaluate_prediction(predictions, actual, model_name):
    errors = (predictions - actual)/actual
    mape =100*(np.abs(errors).mean())
    print(model_name + ':')
    print('Mean Absolute Percentage Error: {:.8f}'.format(mape))
    print('')
evaluate_prediction(prediction_lstm, y_test, 'LSTM')
evaluate_prediction(prediction_bilstm, y_test, 'Bidirectiona LSTM')
evaluate_prediction(prediction_gru, y_test, 'GRU')
evaluate_prediction(prediction_bigru, y_test, 'Bidirectiona GRU')


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




