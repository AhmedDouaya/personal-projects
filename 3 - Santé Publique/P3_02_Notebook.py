#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.cluster import KMeans


# In[2]:


data = pd.read_csv('Data_food.csv', sep='\t')
data.head()


# In[3]:


data['product_name'].value_counts()


# In[4]:


data['pnns_groups_2'].value_counts()


# In[5]:


data['nova_group'].value_counts()


# In[6]:


df1 = data
percent_missing = df1.isnull().mean(axis = 0) * 100
print(percent_missing)


# In[7]:


df2 = df1.append(percent_missing, ignore_index = True)
df2.tail()


# In[8]:


pourcentage = df2.iloc[-1,:]

pourcentages = pourcentage.to_frame()
pourcentages.columns = ['Cat']
display(pourcentages)


# In[9]:


rslt_df = pourcentages.loc[pourcentages['Cat'] < 60]
display(rslt_df)


# In[10]:


plt.close()
rslt_df.plot(kind="bar", figsize=(18, 9))
plt.title("Pourcentage de données manquantes par colonne")
ax = plt.axes()
ax = ax.set(xlabel='Catégories', ylabel='Nan%')


# In[11]:


df = pd.DataFrame(df1, columns=['product_name', 'pnns_groups_1', 'pnns_groups_2', 'countries','stores', 'nutriscore_score', 'nutriscore_grade', 'energy_100g', 'fat_100g', 'saturated-fat_100g', 'fiber_100g' , 'sugars_100g', 'sodium_100g',
                                'proteins_100g', 'nova_group'])
display(df)


# In[12]:


appellations_fr = ['France', 'france', 'en:fr', 'en:france', 'فرنسا','Frankreich', 'Francia', 'fr','法国']
df_france = df[df['countries'].isin(appellations_fr)]
display(df_france)


# In[13]:


df_final = df_france.copy()

cols1 = ['fat_100g','fiber_100g','sugars_100g','sodium_100g','proteins_100g']

for col in cols1 : 
    ind1 = df_final.loc[(df_france[col] < 0) | (df_final[col] > 100)].index.to_list()
    df_final.drop(index=ind1, inplace=True)
    
cols2 = ['energy_100g']

for col in cols2 : 
    ind2 = df_final.loc[(df_final[col] < 0) | (df_final[col] > 900)].index.to_list()
    df_final.drop(index=ind2, inplace=True)

cols3 = ['nova_group']

for col in cols3 : 
    ind3 = df_final.loc[(df_final[col] == 1.0)].index.to_list()
    df_final.drop(index=ind3, inplace=True)
    
display(df_final)


# In[14]:


df_clean = df_final.copy()
df_clean.dropna(subset=['product_name'], inplace=True)

display(df_clean)


# In[15]:


df_clean['nova_group'].value_counts()


# In[16]:


df_clean['pnns_groups_2'].value_counts()


# In[17]:


df_clean.describe()


# In[18]:


percent_missing2 = df_clean.isnull().mean(axis = 0) * 100
print(percent_missing2)


# In[19]:


plt.plot(df_clean['nutriscore_score'], df_clean['sugars_100g'], 'ro', markersize= 1)
plt.show()


# In[20]:


df_clean[(df_clean['fat_100g'] > 10) & (df_clean['nutriscore_score'] < 0)]


# sns.pairplot(
#     df_clean,
#     x_vars=['energy_100g','fat_100g','fiber_100g','sugars_100g','salt_100g','sodium_100g','proteins_100g'],
#     y_vars=['energy_100g','fat_100g','fiber_100g','sugars_100g','salt_100g','sodium_100g','proteins_100g'], hue='nutriscore_grade'
# )
# 
# pass

# In[21]:


print('Nombre de produits :', len(df_clean['product_name']))


# In[22]:


display(df_clean['product_name'])


# In[23]:


df_clean['product_name'].value_counts()


# In[24]:


produits = {"escalope de poulet" : "Escalopes de poulet",
    "aiguillette de poulet" : "Aiguillettes de poulet",
    "blanc de poulet" : "Blancs de poulet",
    "filet de poulet" : "Filets de poulet",
    "escalope de dinde" : "Escalopes de dinde",
    "aiguillette de dinde" : "Aiguillettes de dinde",
    "blanc de dinde" : "Blancs de dinde",
    "aiguillette de canard" : "Aiguillettes de canard",
    "Ice Cream" : "Crème Glacée",
    "Banane" : "Bananes",
    "Soupe de poisson" : "Soupe de poissons"}


# In[25]:


for key, value in produits.items():
    df_clean["product_name"].loc[df_clean["product_name"].str.contains(key, case=False, regex=False)] = value


# In[26]:


df_fr = df_clean.copy()
df_fr.dropna(subset=['nutriscore_score'], inplace=True)

display(df_fr)


# In[27]:


plt.figure(figsize=(10, 5))
sns.countplot(y="product_name", data=df_fr, order=pd.value_counts(df_fr['product_name']).iloc[:20].index)
plt.ylabel("Produits")
plt.show()


# In[28]:


plt.figure(figsize=(7, 5))
sns.countplot(y="pnns_groups_2", data=df_fr, order=pd.value_counts(df_fr['pnns_groups_2']).iloc[:20].index)
plt.ylabel("Groupe de Produits")
plt.show()


# In[29]:


sns.countplot(df_fr['nutriscore_grade'], data=df_fr)
plt.show()


# In[30]:


plt.figure(figsize=(20, 7))
sns.countplot(df_fr['nutriscore_score'], data=df_fr)
plt.show()


# In[31]:


from pivottablejs import pivot_ui

pivot_ui(df_fr)


# In[32]:


corr_matrix = df_fr.corr()
mask = np.triu(corr_matrix)

plt.figure(figsize=(15,15))
sns.heatmap(corr_matrix, cmap="coolwarm_r", mask=mask, linewidths=.5, annot=True, cbar=True, square=True)
plt.show()


# Uniformiser les valeurs selon le pnns group 2 ?
# - Tentative annulée
# 
# for cols in df_clean.columns:
#     if df_clean[cols].dtypes == "float64":
#         df_clean[cols].fillna(df_clean.groupby('pnns_groups_2')[cols].transform('median'), inplace=True)
#         
# display(df_clean)

# sns.lmplot(x="fat_100g", y="nutriscore_score", data=df_fr)
# plt.title("Régression nutriscore/graisse")
# plt.xlim(df_fr["fat_100g"].min(), df_fr["fat_100g"].max())
# plt.ylim(df_fr["nutriscore_score"].min(), df_fr["nutriscore_score"].max())
# plt.show()

# sns.lmplot(x="sugars_100g", y="nutriscore_score", data=df_fr)
# plt.title("Régression nutriscore/sucre")
# plt.xlim(df_fr["sugars_100g"].min(), df_fr["sugars_100g"].max())
# plt.ylim(df_fr["nutriscore_score"].min(), df_fr["nutriscore_score"].max())
# plt.show()

# sns.lmplot(x="energy_100g", y="nutriscore_score", data=df_fr)
# plt.title("Régression nutriscore/energie")
# plt.xlim(df_fr["energy_100g"].min(), df_fr["energy_100g"].max())
# plt.ylim(df_fr["nutriscore_score"].min(), df_fr["nutriscore_score"].max())
# plt.show()

# sns.lmplot(x="proteins_100g", y="nutriscore_score", data=df_clean)
# plt.title("Régression nutriscore/protéines")
# plt.xlim(df_fr["proteins_100g"].min(), df_fr["proteins_100g"].max())
# plt.ylim(df_fr["nutriscore_score"].min(), df_fr["nutriscore_score"].max())
# plt.show()

# sns.lmplot(x="fiber_100g", y="nutriscore_score", data=df_fr)
# plt.title("Régression nutriscore/fibres")
# plt.xlim(df_fr["fiber_100g"].min(), df_fr["fiber_100g"].max())
# plt.ylim(df_fr["nutriscore_score"].min(), df_fr["nutriscore_score"].max())
# plt.show()

# sns.lmplot(x="sodium_100g", y="nutriscore_score", data=df_fr)
# plt.title("Régression nutriscore/sodium")
# plt.xlim(df_fr["sodium_100g"].min(), df_fr["sodium_100g"].max())
# plt.ylim(df_fr["nutriscore_score"].min(), df_fr["nutriscore_score"].max())
# plt.show()

# sns.lmplot(x="saturated-fat_100g", y="nutriscore_score", data=df_fr)
# plt.title("Régression nutriscore/graisses sat")
# plt.xlim(df_fr["saturated-fat_100g"].min(), df_fr["saturated-fat_100g"].max())
# plt.ylim(df_fr["nutriscore_score"].min(), df_fr["nutriscore_score"].max())
# plt.show()

# In[33]:


df_test = df_fr.copy()
df_test = df_test.dropna()


# In[34]:


df_test.replace([np.inf, -np.inf], np.nan, inplace=True)


# In[35]:


df_test = df_test.reset_index()


# In[36]:


X = df_test[['fat_100g','fiber_100g','sugars_100g','sodium_100g','proteins_100g', 'nova_group','energy_100g', 'saturated-fat_100g']]
X


# In[37]:


y = df_test[['nutriscore_score']]
y


# In[38]:


regr = linear_model.LinearRegression()
regr.fit(X, y)


# In[39]:


model = RandomForestRegressor()
model.fit(X, y)


# In[40]:


y_pred2 = model.predict(X)


# In[41]:


plt.close()
sns.scatterplot(y.values[:,0], y_pred2)


# In[42]:


regr.predict(X.iloc[:5])


# In[43]:


np.linspace(3,6,100)


# In[44]:


np.arange(3,6,0.1)


# In[45]:


regr.coef_
{n:v for n,v in zip(X.columns, regr.coef_[0])} 


# In[46]:


regr.intercept_


# In[47]:


regr.n_features_in_


# In[48]:


regr.score(X,y)


# In[49]:


y_pred = regr.predict(X)
y_pred


# In[50]:


y.values


# In[59]:


plt.close()
sns.scatterplot(y.values[:,0], y_pred[:,0])


# In[52]:


get_ipython().run_line_magic('matplotlib', 'notebook')


# In[53]:


from sklearn.model_selection import train_test_split
target = y

xtrain, xtest, ytrain, ytest = train_test_split(X, target, train_size=0.8)


# In[58]:


plt.close()
from sklearn import neighbors

knn = neighbors.KNeighborsClassifier(n_neighbors=1)
knn.fit(xtrain, np.ravel(ytrain,order='C'))
ytest1 = knn.predict(xtest)
sns.scatterplot(ytest.values[:,0], ytest1)


# In[55]:


error = 1 - knn.score(xtest, ytest)
print('Erreur: %f' % error)


# In[61]:


plt.close()
errors = []
for k in range(1,10):
    knn = neighbors.KNeighborsClassifier(k)
    errors.append(100*(1 - knn.fit(xtrain, ytrain.values.ravel()).score(xtest, ytest.values.ravel())))
plt.plot(range(1,10), errors, 'o-')
plt.show()


# In[57]:


df_test.to_csv('out.csv')

