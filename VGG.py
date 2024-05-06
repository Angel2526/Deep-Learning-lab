#!/usr/bin/env python
# coding: utf-8

# In[33]:


import tensorflow as tf
from tensorflow.keras import models, layers
import numpy as np
from tensorflow import keras
from keras.datasets import cifar10
import pandas as pd


# In[34]:


(X_train, y_train), (X_test, y_test) = cifar10.load_data()


# In[35]:


X_train


# In[36]:


y_train


# In[37]:


len(np.unique(y_train))


# In[38]:


X_train.shape


# In[39]:


X_test.shape


# In[40]:


from keras.utils import to_categorical


# In[42]:


X_train = X_train.astype('float32')
X_test = X_test.astype('float32')


# In[43]:


X_train =X_train / 255
X_test = X_test / 255


# In[46]:


y_train = to_categorical(y_train,num_classes = 10)
y_test = to_categorical(y_test,num_classes = 10)


# In[47]:


X_train


# In[48]:


X_train[0]


# In[49]:


import matplotlib.pyplot as plt
plt.figure(figsize=(10,10))
for i in range(12):
    plt.subplot(4, 3, i + 1)
    plt.imshow(X_train[i],cmap='gray')
 
    plt.axis('off')

plt.show()


# In[50]:


from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input


# In[51]:


base_model = VGG16(include_top=False,weights='imagenet',input_shape=(32,32,3))
base_model.trainable = False


# In[52]:


X_train = preprocess_input(X_train)
X_test = preprocess_input(X_test)


# In[53]:


base_model.summary()


# In[54]:


from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras import models
from keras.models import Sequential
flatten_layer = Flatten()
dense_layer_1 = Dense(50, activation='relu')
dense_layer_2 = Dense(20, activation='relu')
prediction_layer = Dense(10, activation='softmax')

model = Sequential([base_model, flatten_layer, dense_layer_1, dense_layer_2, prediction_layer])


# In[55]:


model.summary()


# In[56]:


model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])


# In[57]:


model.fit(X_train, y_train, epochs=1, validation_split = 0.2)


# In[22]:


model.evaluate(X_test, y_test)


# In[23]:


model.predict(X_test)


# In[ ]:


import matplotlib.pyplot as plt
plt.plot(history.history['accuracy'],label='Training_Accuracy')
plt.plot(history.history['val_accuracy'],label='Validation_Accuracy')

plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend();


# # using mnist
# 

# In[59]:


import tensorflow as tf
from keras.datasets import mnist
(X_train, y_train), (X_test, y_test) = mnist.load_data()


# In[60]:


X_train[0].shape


# In[61]:


import numpy as np
X_train = np.repeat(tf.image.resize(X_train[..., np.newaxis], (32, 32)).numpy(), 3, axis=-1)
X_test = np.repeat(tf.image.resize(X_test[..., np.newaxis], (32, 32)).numpy(), 3, axis=-1)


# In[62]:


from tensorflow.keras.utils import to_categorical
y_train = to_categorical(y_train, num_classes=10)
y_test = to_categorical(y_test, num_classes=10)


# In[63]:


X_train[0].shape


# In[69]:


import matplotlib.pyplot as plt
#plt.figure(figsize=(10,10))
for i in range(12):
    plt.subplot(4, 3, i + 1)
    plt.imshow(X_train[i],cmap='gray')
 
    plt.axis('off')

plt.show()


# In[70]:


from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(32,32,3))
base_model.trainable = False

X_train = preprocess_input(X_train)
X_test = preprocess_input(X_test)


# In[71]:


base_model.summary()


# In[72]:


from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras import models

flatten_layer = Flatten()
dense_layer_1 = Dense(50, activation='relu')
dense_layer_2 = Dense(20, activation='relu')
prediction_layer = Dense(10, activation='softmax')

model = Sequential([base_model, flatten_layer, dense_layer_1, dense_layer_2, prediction_layer])


# In[73]:


model.summary()


# In[74]:


model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])


# In[75]:


history = model.fit(X_train, y_train, epochs=3, validation_split =0.2)


# In[76]:


model.evaluate(X_test, y_test)


# In[77]:


import matplotlib.pyplot as plt
plt.plot(history.history['accuracy'],label='Training_Accuracy')
plt.plot(history.history['val_accuracy'],label='Validation_Accuracy')

plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend();


# In[78]:


import matplotlib.pyplot as plt
plt.plot(history.history['loss'],label='Training_loss')
plt.plot(history.history['val_loss'],label='Validation_loss')

plt.xlabel("Epochs")
plt.ylabel("loss")
plt.legend();


# In[ ]:




