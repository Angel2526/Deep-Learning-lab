#!/usr/bin/env python
# coding: utf-8

# In[15]:


# importing the hand written digit dataset
from sklearn import datasets

# digit contain the dataset
digits = datasets.load_digits()

# dir function use to display the attributes of the dataset
dir(digits)


# In[16]:


# outputting the picture value as a series of numbers
print(digits.images[0])


# In[17]:


# importing the matplotlib libraries pyplot function
import matplotlib.pyplot as plt
# defining the function plot_multi

def plot_multi(i):
	nplots = 16
	fig = plt.figure(figsize=(15, 15))
	for j in range(nplots):
		plt.subplot(4, 4, j+1)
		plt.imshow(digits.images[i+j], cmap='binary')
		plt.title(digits.target[i+j])
		plt.axis('off')
	


# In[18]:


# converting the 2 dimensional array to one dimensional array
y = digits.target
x = digits.images.reshape((len(digits.images), -1))

# gives the shape of the data
x.shape


# In[19]:


# printing the one-dimensional array's values
x[0]


# In[20]:


# Very first 1000 photographs and
# labels will be used in training.
x_train = x[:1000]
y_train = y[:1000]

# The leftover dataset will be utilised to
# test the network's performance later on.
x_test = x[1000:]
y_test = y[1000:]


# In[21]:


# importing the MLP classifier from sklearn
from sklearn.neural_network import MLPClassifier

# calling the MLP classifier with specific parameters
mlp = MLPClassifier(hidden_layer_sizes=(15,),
					activation='logistic',
					alpha=1e-4, solver='sgd',
					tol=1e-4, random_state=1,
					learning_rate_init=.1,
					verbose=True)


# In[25]:


plt.show()

plot_multi(0)


# In[10]:


mlp.fit(x_train, y_train)


# In[11]:


fig, axes = plt.subplots(1, 1)
axes.plot(mlp.loss_curve_, 'o-')
axes.set_xlabel("number of iteration")
axes.set_ylabel("loss")
plt.show()


# In[12]:


predictions = mlp.predict(x_test)
predictions[:50]


# In[13]:


y_test[:50]


# In[14]:


# importing the accuracy_score from the sklearn
from sklearn.metrics import accuracy_score

# calculating the accuracy with y_test and predictions
accuracy_score(y_test, predictions)


# In[ ]:





# In[ ]:




