#!/usr/bin/env python
# coding: utf-8

# # USING SUNSPOTS

# In[1]:


from pandas import read_csv
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, SimpleRNN
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import pandas as pd


# In[2]:


Data = pd.read_csv("C:/Users/ANGEL SARA PETER/Downloads/Sunspots.csv",index_col ="Date",parse_dates =True)


# In[3]:


Data


# In[4]:


Data.head()


# In[5]:


Data.tail()


# In[6]:


Data.shape


# In[7]:


Data.info()


# In[8]:


Data.isnull().sum()


# In[9]:


Data.nunique()


# In[10]:


scaler = MinMaxScaler(feature_range=(0, 1))
Data = scaler.fit_transform(Data).flatten()


# In[11]:


Data


# In[12]:


Train =Data[:2500]
Test = Data[2500:]


# In[13]:


Train


# In[14]:


Test


# In[15]:


from keras.preprocessing.sequence import TimeseriesGenerator


# In[16]:


n_input = 3
n_features = 1
generator = TimeseriesGenerator(Train, Train, length=n_input, batch_size=1)
generatorTest=TimeseriesGenerator(Test,Test,length=n_input,batch_size=1)


# In[17]:


generator[0]


# In[18]:


generatorTest[0]


# In[19]:


from keras.layers import Dense,SimpleRNN,LSTM,GRU


# In[41]:


modelSunSpots = Sequential()
modelSunSpots.add(SimpleRNN(units=100, input_shape=(n_input,n_features),activation='tanh'))
modelSunSpots.add(Dense(n_features,activation='tanh'))


# In[42]:


modelSunSpots.compile(loss='mean_squared_error', optimizer='adam',metrics =['accuracy'])


# In[43]:


modelSunSpots.fit(generator,epochs=3)


# In[44]:


modelSunSpots.evaluate(generatorTest)


# In[48]:


Predictions=modelSunSpots.predict(generatorTest[5][0])


# In[51]:


print("Actual Input:\n",generatorTest[5][0])
print("Actual Output:\n",generatorTest[5][1])
print("Precidicted using SimpleRNN:\n",Predictions)


# # using NIFTY50

# In[1]:


from pandas import read_csv
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, SimpleRNN
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import pandas as pd


# In[2]:


Data = pd.read_csv("C:/Users/ANGEL SARA PETER/Downloads/NIFTY.csv",index_col ="Date",parse_dates =True)


# In[3]:


Data


# In[4]:


Data.head()


# In[5]:


Data.tail()


# In[6]:


Data.shape


# In[7]:


scaler = MinMaxScaler(feature_range=(0, 1))
Data = scaler.fit_transform(Data)#flatten()


# In[8]:


Train =Data[:1500]
Test = Data[1500:]


# In[9]:


Train


# In[10]:


from keras.preprocessing.sequence import TimeseriesGenerator


# In[28]:


n_input = 10
n_features = 4
generator = TimeseriesGenerator(Train, Train, length=n_input, batch_size=1)
generatorTest=TimeseriesGenerator(Test,Test,length=n_input,batch_size=1)


# In[29]:


from keras.layers import Dense,SimpleRNN,LSTM,GRU


# In[30]:


#SimpleRNN
modelnifty = Sequential()
modelnifty.add(SimpleRNN(units=100, input_shape=(n_input,n_features),activation='tanh'))
modelnifty.add(Dense(n_features,activation='tanh'))


# In[31]:


modelnifty.compile(loss='mean_squared_error', optimizer='adam',metrics =['accuracy'])


# In[32]:


history= modelnifty.fit(generator,epochs=3)


# In[33]:


modelnifty.evaluate(generatorTest)


# In[34]:


modelnifty.predict(generatorTest)


# In[18]:


generatorTest[1]


# In[19]:


generatorTest[10][0]


# In[22]:


#LSTM
model_l=Sequential()
model_l.add(LSTM(units=100,input_shape=(n_input,n_features),activation="tanh"))
model_l.add(Dense(n_features,activation="tanh"))
model_l.compile(optimizer='adam',loss="mean_squared_error",metrics=['accuracy'])
history_l=model_l.fit(generator,epochs=5)
loss_l,accuracy_l=model_l.evaluate(generatorTest)


# In[23]:


model_l.predict(generatorTest)


# In[25]:


#GRU
model_g=Sequential()
model_g.add(GRU(units=100,input_shape=(n_input,n_features),activation="tanh"))
model_g.add(Dense(n_features,activation="tanh"))
model_g.compile(optimizer='adam',loss="mean_squared_error",metrics=['accuracy'])
history_g=model_g.fit(generator,epochs=3)
loss_g,accuracy_g=model_g.evaluate(generatorTest)


# In[35]:


plt.plot(history.history['accuracy'],label="SimpleRNN")
plt.plot(history_l.history['accuracy'],label="LSTM")
plt.plot(history_g.history['accuracy'],label="GRU")

plt.title('Training Accuracy')
plt.xlabel("epochs")
plt.ylabel("accuracy")
plt.legend()


# In[36]:


plt.plot(history.history['loss'],label="SimpleRNN")
plt.plot(history_l.history['loss'],label="LSTM")
plt.plot(history_g.history['loss'],label="GRU")
plt.title('Training Loss')
plt.xlabel("epochs")
plt.ylabel("loss")
plt.legend()


# In[39]:


prediction_s=modelnifty.predict(generatorTest[10][0])
prediction_l=model_l.predict(generatorTest[10][0])
prediction_g=model_g.predict(generatorTest[10][0])


# In[40]:


print("Actual Input:",generatorTest[10][0])
print("Actual Output:",generatorTest[10][1])
print("Precidicted using SimpleRNN:",prediction_s)
print("Precidicted using LSTM:     ",prediction_l)
print("Precidicted using GRU:      ",prediction_g)


# In[ ]:




