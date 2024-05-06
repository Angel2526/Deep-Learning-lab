#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf
import numpy as np
from tensorflow import keras
from tensorflow.keras.layers import Dense, Flatten
from keras import layers,models
from keras.utils import to_categorical


# In[2]:


from keras.models import Sequential
import matplotlib.pyplot as plt


# In[3]:


from keras.datasets import mnist


# In[4]:


(X_train, y_train), (X_test, y_test) = mnist.load_data()


# In[5]:


X_train.shape


# In[6]:


X_train = X_train.astype('float32')
X_test = X_test.astype('float32')


# In[7]:


X_train =X_train / 255
X_test = X_test / 255


# In[8]:


X_train


# In[9]:


X_train[0]


# In[10]:


y_train = to_categorical(y_train, num_classes = 10)
y_test = to_categorical(y_test, num_classes = 10)


# In[11]:


y_train 


# In[12]:


y_train[0]


# In[15]:


import matplotlib.pyplot as plt
plt.figure(figsize=(10,10))
for i in range(9):
    plt.subplot(3, 3, i + 1)
    plt.imshow(X_train[i],cmap='gray')
 
    plt.axis('off')

plt.show()


# In[19]:


model = Sequential()
model.add(Flatten(input_shape=(28,28,1)))
model.add(Dense(512,activation = 'relu'))
model.add(Dense(256,activation = 'relu'))
model.add(Dense(128,activation = 'relu'))
model.add(Dense(10,activation = 'softmax'))


# In[20]:


model.summary()


# In[21]:


model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])


# In[22]:


history = model.fit(X_train, y_train, epochs = 3,validation_split=0.2)


# In[23]:


model.evaluate(X_test, y_test)


# In[24]:


model.predict(X_test)


# In[25]:


import matplotlib.pyplot as plt
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()


# In[26]:


import matplotlib.pyplot as plt
plt.plot(history.history['loss'],label='Training_loss')
plt.plot(history.history['val_loss'],label='Validation_loss')

plt.xlabel("Epochs")
plt.ylabel("loss")
plt.legend();


# # using CIFAR10

# In[27]:


import tensorflow as tf
import numpy as np
from tensorflow import keras
from tensorflow.keras.layers import Dense, Flatten
from keras import layers,models
from keras.utils import to_categorical
from keras.models import Sequential
import matplotlib.pyplot as plt
from keras.datasets import cifar10


# In[28]:


(X_train, y_train), (X_test, y_test) = cifar10.load_data()


# In[29]:


X_train.shape


# In[30]:


y_train = to_categorical(y_train, num_classes = 10)
y_test = to_categorical(y_test, num_classes = 10)


# In[31]:


X_train =X_train / 255
X_test = X_test / 255


# In[32]:


import matplotlib.pyplot as plt
plt.figure(figsize=(10,10))
for i in range(12):
    plt.subplot(4,3, i + 1)
    plt.imshow(X_train[i],cmap='gray')
 
    plt.axis('off')

plt.show()


# In[37]:


model = Sequential()
model.add(Flatten(input_shape=(32,32,3)))
model.add(Dense(512,activation = 'relu'))
model.add(Dense(256,activation = 'relu'))
model.add(Dense(128,activation = 'relu'))
model.add(Dense(10,activation = 'softmax'))


# In[38]:


model.summary()


# In[39]:


model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])


# In[40]:


history = model.fit(X_train, y_train, epochs = 3,validation_split=0.2)


# In[41]:


model.evaluate(X_test, y_test)


# In[42]:


import matplotlib.pyplot as plt
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()


# In[43]:


import matplotlib.pyplot as plt
plt.plot(history.history['loss'],label='Training_loss')
plt.plot(history.history['val_loss'],label='Validation_loss')

plt.xlabel("Epochs")
plt.ylabel("loss")
plt.legend();


# In[ ]:




