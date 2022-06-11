#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
import keras
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns


# In[3]:


data = pd.read_csv("E:/THESIS DATASET OF MS-AI/creditcard.csv")


# In[4]:


data


# In[5]:


data.head()


# In[6]:


data.tail(3)


# In[7]:


data[20000:20005]


# In[8]:


len(data)


# In[9]:


data.shape


# In[10]:


data.ndim


# In[11]:


data.columns


# In[12]:


data.info()


# In[13]:


data.describe()


# In[14]:


data.isnull()


# In[15]:


data.isnull().any()


# In[16]:


data.isnull().sum()


# In[17]:


data.nunique()


# In[18]:


data["Amount"]


# In[19]:


data["Amount"].unique()


# In[20]:


len(data["Amount"].unique())


# In[21]:


data["Time"].unique()


# In[22]:


data["Time"].nunique()


# In[23]:


data["Class"].unique()


# In[24]:


data["Class"].nunique()


# In[25]:


data.loc[100] # display row


# In[26]:


data.sample(5)  # display random data


# In[ ]:





# In[33]:


vc = data['Class'].value_counts().to_frame().reset_index()
vc['percent'] = vc["Class"].apply(lambda x : round(100*float(x) / len(data), 2))
vc = vc.rename(columns = {"index" : "Target", "Class" : "Count"})
vc


# In[ ]:





# In[ ]:





# # Visualize Data

# # Display correlation heatmaps

# In[6]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

data = pd.read_csv("E:/THESIS DATASET OF MS-AI/creditcard.csv")


# In[7]:


# Correlation matrix
corr = data.corr() 
plt.figure(figsize=(12,10))
sns.heatmap(data=corr, annot=True, cmap='Spectral').set(title="Correlation Matrix")


# In[ ]:





# In[27]:


fig, ax = plt.subplots(figsize=(20,10))
sns.heatmap(data.corr(),annot=True).set_title('Correlation heatmap without generated features')


# In[32]:


#Determine the number of fraud and valid transactions in the entire dataset
#LABELS = (xlabel("Class") , ylabel("Frequency"))
count_classes = pd.value_counts(data['Class'], sort = True)
count_classes.plot(kind = 'bar', rot=0)
plt.title("Transaction Class Distribution")
plt.xticks(range(2))
plt.xlabel("Class")
plt.ylabel("Frequency")


# In[ ]:





# In[42]:


corelation = data.corr()


# In[47]:


corr= data.corr()
plt.figure(figsize =(12,10))

sns.heatmap(corr,vmax=.8, linewidths = 0.01, square = True, cmap = 'Greens', linecolor = 'black' )
plt.title('corelation between features')
plt.show()


# In[48]:


sns.scatterplot(data.Amount, data.Class)
plt.show()


# In[51]:


x = data['Amount']
y = data['Class']
x1 =  (x,y)
plt.hist(x1)
plt.show() 


# In[ ]:


import seaborn as sns

sns.factorplot('Class', data = data, kind = 'count')
plt.show()


# In[ ]:




