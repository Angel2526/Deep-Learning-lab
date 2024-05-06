#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf
from tensorflow.keras import models, layers
import numpy as np
from tensorflow import keras
from keras.datasets import mnist
import pandas as pd


# In[2]:


(X_train, y_train), (X_test, y_test) = mnist.load_data()


# In[3]:


X_train


# In[4]:


X_train.shape


# In[5]:


X_train[0].shape


# In[6]:


y_train


# In[7]:


num_classes = len(np.unique(y_train))


# In[8]:


num_classes


# In[9]:


from keras.utils import to_categorical


# In[10]:


y_train = to_categorical(y_train, num_classes = 10)
y_test = to_categorical(y_test, num_classes = 10)


# In[11]:


len(y_test)


# In[12]:


y_train


# In[13]:


X_train = X_train.astype('float32')
X_test = X_test.astype('float32')


# In[14]:


X_train


# In[15]:


X_train =X_train / 255
X_test = X_test / 255


# In[16]:


X_train


# In[17]:


X_train=X_train.reshape(60000,28,28,1)
X_test=X_test.reshape(10000,28,28,1)


# In[18]:


X_train


# In[19]:


X_train.shape


# In[20]:


import matplotlib.pyplot as plt


# In[21]:


for i in range(5):
    plt.subplot(1, 5, i + 1)
    plt.imshow(X_train[i],cmap='gray')
 
    plt.axis('off')

plt.show()


# In[22]:


plt.figure(figsize =(10,10))
for i in range(9):
    plt.subplot(3, 3, i + 1)
    plt.imshow(X_train[i],cmap='gray')
 
    plt.axis('off')

plt.show()


# In[23]:


from keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten


# In[24]:


model = Sequential()
model.add(Conv2D(filters = 25, kernel_size = (3,3), padding = 'same', activation = 'relu', input_shape = (28, 28, 1)))
model.add(MaxPooling2D((2, 2)))

model.add(Conv2D(filters = 64, kernel_size = (3,3), padding = 'same', activation = 'relu'))
model.add(MaxPooling2D((2, 2)))

model.add(Conv2D(filters = 64, kernel_size = (3,3), padding = 'same', activation = 'relu'))
model.add(MaxPooling2D((2, 2)))

model.add(Flatten())
model.add(Dense(64, activation = 'relu'))
model.add(Dense(10, activation = 'softmax'))


# In[25]:


model.summary()


# In[26]:


model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])


# In[27]:


history = model.fit(X_train, y_train, epochs = 3,validation_split=0.2)


# In[28]:


model.evaluate(X_test, y_test)


# In[29]:


y_predicted_by_model = model.predict(X_test)


# In[30]:


y_predicted_by_model[0]


# In[31]:


y_predicted_by_model[3]


# In[34]:


import matplotlib.pyplot as plt
plt.plot(history.history['accuracy'],label='Training_Accuracy')
plt.plot(history.history['val_accuracy'],label='Validation_Accuracy')

plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()


# In[33]:


import matplotlib.pyplot as plt
plt.plot(history.history['loss'],label='Training_loss')
plt.plot(history.history['val_loss'],label='Validation_loss')

plt.xlabel("Epochs")
plt.ylabel("loss")
plt.legend();
plt.show()


# In[ ]:




