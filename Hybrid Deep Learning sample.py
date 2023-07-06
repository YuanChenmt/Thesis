#!/usr/bin/env python
# coding: utf-8

# In[1]:


# import kappa 
import pandas as pd
import numpy as np
df=pd.read_excel(DIV1f.xlsx')

print(df)


# In[2]:





# In[3]:


# split data set
train=df.iloc[:40]
test=df.iloc[40:]


# In[4]:


# normalize
from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler()
scaler.fit(train)
scaled_train = scaler.transform(train)
scaled_test = scaler.transform(test)


# In[5]:


from keras.preprocessing.sequence import TimeseriesGenerator


# In[6]:


n_input=30
n_feature=1
generator=TimeseriesGenerator(scaled_train,scaled_train,length=n_input,batch_size=32)


# In[7]:


X,Y=generator[0]


# In[8]:


print(X.shape)
print(Y.shape)


# In[9]:


import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout,GRU, Bidirectional
def lstm(units):
    model = Sequential()
    model.add(LSTM(units,activation='selu',input_shape=(n_input,n_feature)))
    model.add(Dropout(0.2))
    
    model.add(Dense(1))
    model.compile(optimizer='adam',
                 loss='mae')
    return model
model_lstm=lstm(64)

def bilstm(units):
    model = Sequential()
    model.add(Bidirectional(LSTM(units,activation='selu'),input_shape=(n_input,n_feature)))
    model.add(Dropout(0.2))
     
    model.add(Dense(1))

    model.compile(optimizer='adam',
                 loss='mae')
    return model
model_bilstm=bilstm(32)

def bigru(units):
    model = Sequential()
    model.add(Bidirectional(GRU(units,activation='selu'),input_shape=(n_input,n_feature)))
    model.add(Dropout(0.2))
    
    model.add(Dense(1))

    model.compile(optimizer='adam',
                 loss='mae')
    return model
model_bigru=bigru(32)



def gru(units):
    model = Sequential()
    model.add(GRU(units,activation='selu',input_shape=(n_input,n_feature)))
    model.add(Dropout(0.2))
    
    model.add(Dense(1))
    model.compile(optimizer='adam',
                 loss='mae')
    return model



model_gru=gru(32)


# In[10]:


def fit_model(model):
    history= model.fit(generator, epochs=50)
    return history

history_lstm=fit_model(model_lstm)
history_bilstm=fit_model(model_bilstm)
history_bigru=fit_model(model_bigru)
history_gru = fit_model(model_gru)


# In[ ]:





# In[ ]:





# In[11]:


last_trained_batch=scaled_train[-30:]
last_trained_batch=last_trained_batch.reshape(1,n_input,n_feature)
print(last_trained_batch.shape)
print(last_trained_batch)


# In[12]:


pre_lstm=model_lstm.predict(last_trained_batch)
pre_bilstm=model_bilstm.predict(last_trained_batch)
pre_bigru=model_bigru.predict(last_trained_batch)
pre_gru=model_gru.predict(last_trained_batch)


# In[13]:


test_predictions_1=[]
first_batch=scaled_train[-n_input:]
current_batch=first_batch.reshape(1,n_input,n_feature)

for i in range(len(test)):
    current_pred=model_lstm.predict(current_batch)[0]
    test_predictions_1.append(current_pred)
    current_batch=np.append(current_batch[:,1:,:],[[current_pred]],axis=1)
                            
test_predictions_2=[]
first_batch=scaled_train[-n_input:]
current_batch=first_batch.reshape(1,n_input,n_feature)

for i in range(len(test)):
    current_pred=model_gru.predict(current_batch)[0]
    test_predictions_2.append(current_pred)
    current_batch=np.append(current_batch[:,1:,:],[[current_pred]],axis=1)  
    
test_predictions_3=[]
first_batch=scaled_train[-n_input:]
current_batch=first_batch.reshape(1,n_input,n_feature)

for i in range(len(test)):
    current_pred=model_bilstm.predict(current_batch)[0]
    test_predictions_3.append(current_pred)
    current_batch=np.append(current_batch[:,1:,:],[[current_pred]],axis=1)                            

    test_predictions_4=[]
first_batch=scaled_train[-n_input:]
current_batch=first_batch.reshape(1,n_input,n_feature)

for i in range(len(test)):
    current_pred=model_bigru.predict(current_batch)[0]
    test_predictions_4.append(current_pred)
    current_batch=np.append(current_batch[:,1:,:],[[current_pred]],axis=1)                            
    
    


# In[14]:


print(test_predictions_1)
print(test_predictions_2)


# In[15]:


true_predictions_lstm=scaler.inverse_transform(test_predictions_1)
true_predictions_gru=scaler.inverse_transform(test_predictions_2)
true_predictions_bilstm=scaler.inverse_transform(test_predictions_3)
true_predictions_bigru=scaler.inverse_transform(test_predictions_4)


# In[16]:


print(true_predictions_lstm)


# In[17]:


print(true_predictions_bilstm)


# In[ ]:


print(true_predictions_bigru)


# In[18]:


print(true_predictions_gru)


# In[19]:





# In[20]:





# In[ ]:





# In[ ]:





# In[ ]:




