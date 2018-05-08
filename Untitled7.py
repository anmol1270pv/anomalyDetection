
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
import matplotlib
import seaborn
from matplotlib import pyplot as plt
from sklearn import preprocessing
from sklearn.cluster import KMeans


# In[3]:


from sklearn.svm import OneClassSVM


# In[4]:


df=pd.read_csv("C:/Users/anmol narang/Documents/artificialWithAnomaly/art_increase_spike_density.csv")


# In[5]:


df.head()


# In[8]:


df['timestamp']=pd.to_datetime(df['timestamp'])


# In[9]:


df['hours']=df['timestamp'].dt.hour


# In[10]:


df['dayOfTheWeek']=df['timestamp'].dt.dayofweek


# In[11]:


df['weekday']=(df['dayOfTheWeek']<5).astype(int)


# In[12]:


outliers_fraction=0.01


# In[13]:


df['time_epoch'] = (df['timestamp'].astype(np.int64)/100000000000).astype(np.int64)


# In[14]:


data=df[['value','hours','dayOfTheWeek','weekday']]


# In[16]:


min_max_scaler=preprocessing.StandardScaler()


# In[17]:


np_scaled=min_max_scaler.fit_transform(data)


# In[18]:


model=OneClassSVM(nu=0.95*outliers_fraction)


# In[19]:


data=pd.DataFrame(np_scaled)


# In[21]:


model.fit(data)
df['anomaly26']=pd.Series(model.predict(data))


# In[22]:


df['anomaly26']=df['anomaly26'].map({1:0,-1:1})


# In[23]:


print(df['anomaly26'].value_counts())


# In[25]:


fig,ax=plt.subplots()
a=df.loc[df['anomaly26']==1,['time_epoch','value']]
ax.plot(df['time_epoch'],df['value'],color='blue')
ax.scatter(a['time_epoch'],a['value'],color='red')
plt.show()


# In[26]:


a = df.loc[df['anomaly26'] == 0, 'value']
b = df.loc[df['anomaly26'] == 1, 'value']

fig, axs = plt.subplots()
axs.hist([a,b], bins=32, stacked=True, color=['blue', 'red'], label=['normal', 'anomaly'])
plt.legend()
plt.show()

