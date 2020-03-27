#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as splt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn import metrics 
from sklearn.metrics import confusion_matrix
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn import svm


# In[2]:


#loading the dataset
original_data = pd.read_csv("bank-full.csv", sep=";")
original_data.head()


# In[3]:


####### Statistical Exploration ########


# In[4]:


original_data.info()


# In[5]:


original_data.describe()
# We observe some extreme maximum values and high standard deviation values
# which will be further investigated


# In[6]:


#Check for duplicates
original_data.duplicated().sum()


# In[7]:


#check for missing values
print(original_data.isnull().sum())


# In[8]:


# Looking through the data points alternativelly marked as missing values
print('\nJobs: ', original_data.job.unique())
print('\nMarital status: ', original_data.marital.unique())
print('\nEducation: ', original_data.education.unique())
print('\nCredit in default: ', original_data.default.unique())
print('\nHousing loan: ', original_data.housing.unique())
print('\nPersonal loan: ', original_data.loan.unique())
print('\nContact channel: ', original_data.contact.unique())
print('\nMonth: ', original_data.month.unique())
print('\nPrevious campaign outcome: ', original_data.poutcome.unique())


# In[9]:


# Potential missing values
print('\nUnknown values in job: ', (original_data.job == 'unknown').sum())
print('\nUnknown values in education: ', (original_data.education == 'unknown').sum())
print('\nUnknown values in contact: ', (original_data.contact == 'unknown').sum())
print('\nUnknown values in poutcome: ', (original_data.poutcome == 'unknown').sum())


# In[10]:


# Dealing with potential missing values
(original_data.pdays == -1).sum()
(original_data.poutcome == 'other').sum()


# In[11]:


# Ensuring there no unnatural values
if(original_data.age.any() < 0 or original_data.age.any() > 100):
    print('\nLikely Impossible!')
else:
    print('\nGood data!')
if(original_data.day.any() < 1 or original_data.age.any() > 31):
    print('\nLikely Impossible!')
else:
    print('\nGood data!')
if(original_data.duration.any() <= 0):
    print('\nLikely Impossible!')
else:
    print('\nGood data!')
if(original_data.campaign.any() <= 0):
    print('\nLikely Impossible!')
else:
    print('\nGood data!')
if(original_data.previous.any() <= 0):
    print('\nLikely Impossible!')
else:
    print('\nGood data!')
if(original_data.pdays.any() < 0 and original_data.pdays.any() != -1):
    print('\nLikely Impossible!')
else:
    print('\nGood data!')


# In[12]:


####### Data Visualization ########


# In[13]:


# Count plot of 'age' feature
sns.set(style = 'darkgrid')
ax = sns.distplot(original_data.age, kde = False, color = '#FF3A03')
ax.set_title('Age Values Count', color = '#0E189C')
ax.set_ylabel('Frequency', color= '#0E189C')
ax.set_xlabel('Age', color = '#0E189C')


# In[14]:


# Count plot of 'balance' feature
# Find how many data points are, to know how many bins to create
balance_array = original_data.balance.unique()
print(len(balance_array))
# Plot histogram
sns.set()
ax1 = sns.distplot(original_data.balance, bins = 84, kde = False, color = '#FF3A03')
ax1.set_title('Balance Values Count', color = '#0E189C')
ax1.set_ylabel('Frequency', color = '#0E189C')
ax1.set_xlabel('Balance($)', color = '#0E189C')


# In[15]:


# Outliers in balance
copy_2 = original_data
copy_1 = copy_2
copy_1.balance = copy_1.balance.where(copy_1.balance < 20000, other = 20000)
copy_1.balance = copy_1.balance.where(copy_1.balance > -2000, other = -2000)
balance_array = copy_1.balance.unique()
print(len(balance_array))
sns.set()
ax = sns.distplot(copy_1.balance, bins = 84, kde = False, color = '#FF2E2E11')
ax.set_title('Balance Values Count', color = '#0E189C')
ax.set_ylabel('Frequency', color = '#0E189C')
ax.set_xlabel('Balance($)', color = '#0E189C')


# In[16]:


sns.set()
ax = sns.boxplot(copy_1.balance, color = '#FF3A03', fliersize = 1, saturation = 1)
ax.set_title('Balance Values Count', color = '#0E189C')
ax.set_ylabel('Frequency', color = '#0E189C')
ax.set_xlabel('Balance($)', color = '#0E189C')


# In[17]:


# Count plot of 'duration' feature
# Find how many data points are, to know how many bins to create
duration_array = original_data.duration.unique()
print(len(duration_array))
# Plot histogram
sns.set(style = 'darkgrid')
ax = sns.distplot(original_data.duration, bins = 40, kde = False, color = '#FF3A03')
ax.set_title('Durations Count', color = '#0E189C')
ax.set_ylabel('Frequency', color= '#0E189C')
ax.set_xlabel('Duration(s)', color = '#0E189C')


# In[18]:


# Outliers in duration
copy_2 = original_data
copy_1 = copy_2
copy_1.duration = copy_1.duration.where(copy_1.duration < 2000, other = 2000)
balance_array = copy_1.duration.unique()
print(len(balance_array))
sns.set()
ax = sns.distplot(copy_1.duration, bins = 40, kde = False, color = '#FF3A03')
ax.set_title('Duration Values Count', color = '#0E189C')
ax.set_ylabel('Frequency', color = '#0E189C')
ax.set_xlabel('Duration(s)', color = '#0E189C')


# In[19]:


sns.set()
ax = sns.boxplot(copy_1.duration, color = '#FF3A03', fliersize = 1, saturation = 1)
ax.set_title('Duration percentiles', color = '#0E189C')
ax.set_ylabel('Frequency', color = '#0E189C')
ax.set_xlabel('Duration(s)', color = '#0E189C')


# In[20]:


sns.set()
ax = sns.boxplot(copy_1.campaign, color = '#FF3A03', fliersize = 1, saturation = 1)
ax.set_title('Campaign Percentiles', color = '#0E189C')
ax.set_ylabel('Frequency', color = '#0E189C')
ax.set_xlabel('campaign', color = '#0E189C')


# In[21]:


copy_1.previous = copy_1.previous.where(copy_1.previous < 50, other = 50)
sns.set()
ax = sns.boxplot(copy_1.previous, color = '#FF3A03', fliersize = 1, saturation = 1)
ax.set_title('Previous Percentiles', color = '#0E189C')
ax.set_ylabel('Frequency', color = '#0E189C')
ax.set_xlabel('Previous', color = '#0E189C')


# In[22]:


sns.set()
ax = sns.boxplot(copy_1.pdays, color = '#FF3A03')
ax.set_title('Pdays Percentiles', color = '#0E189C')
ax.set_ylabel('Frequency', color = '#0E189C')
ax.set_xlabel('Pdays', color = '#0E189C')


# In[23]:


pie_array = np.zeros(4, dtype = int)
pie_array[0] = sum(original_data.poutcome == 'unknown')
pie_array[1] = sum(original_data.poutcome == 'other')
pie_array[2] = sum(original_data.poutcome == 'success')
pie_array[3] = sum(original_data.poutcome == 'failure')
print(pie_array[::])
splt.pie(pie_array, radius = 2, labels = ['unknown', 'other', 'success', 'failure'], explode = [0.1, 0.1, 0.1, 0.1])


# In[24]:


pie_array = np.zeros(4, dtype = int)
pie_array[0] = sum(original_data.education == 'unknown')
pie_array[1] = sum(original_data.education == 'primary')
pie_array[2] = sum(original_data.education == 'secondary')
pie_array[3] = sum(original_data.education == 'tertiary')
print(pie_array[::])
splt.pie(pie_array, radius = 2, labels = ['unknown', 'primary', 'secondary', 'tertiary'], explode = [0.05, 0.05, 0.05, 0.05])
splt.title('Education repartition')


# In[25]:


sns.set()
splt.figure(figsize = [14,8])
ax = sns.violinplot(y=original_data.balance, x=original_data.job, saturation = 1)


# In[26]:


# we plot the scatter matrix plot to find relation 
#pd.plotting.scatter_matrix(data)


# In[27]:


#splt.matshow(original_data.corr(method ='pearson'))
corr =original_data.corr(method = 'pearson')
sns.heatmap(corr)


# In[ ]:





# In[28]:


#data transformation. From categorical to numerical value
original_data = pd.read_csv("bank-full.csv", sep=";")

le = preprocessing.LabelEncoder()
le.fit(original_data['job'])
original_data['job'] = (le.transform(original_data['job']))
x = [x for x in range(len(set(original_data['job'])))]
print(x,le.inverse_transform(x) )

le.fit(original_data['marital'])
original_data['marital'] = (le.transform(original_data['marital']))

le.fit(original_data['education'])
original_data['education'] = (le.transform(original_data['education']))

le.fit(original_data['default'])
original_data['default'] = (le.transform(original_data['default']))

le.fit(original_data['housing'])
original_data['housing'] = (le.transform(original_data['housing']))

le.fit(original_data['loan'])
original_data['loan'] = (le.transform(original_data['loan']))

le.fit(original_data['contact'])
original_data['contact'] = (le.transform(original_data['contact']))

le.fit(original_data['month'])
original_data['month'] = (le.transform(original_data['month']))

le.fit(original_data['poutcome'])
original_data['poutcome'] = (le.transform(original_data['poutcome']))

le.fit(original_data['y'])
original_data['y'] = (le.transform(original_data['y']))


# In[29]:


###data mining and descriptive

print(original_data.head(),"\n",  original_data.describe())

### we want to know the repartition of the class
print(original_data.y.value_counts())

print("proportion of classes:\n", original_data.y.value_counts()/len(original_data.y))

#### here we need to plot histogram to see if there are clear boundaries to choose variable
### We need to find if there are outliers too


# In[30]:


x = original_data.iloc[:,:-1]
y = original_data.iloc[:,-1]
print(x.head(),"\n" ,y.head())


# In[31]:


corr =original_data.corr(method = 'pearson')
sns.heatmap(corr)


# In[33]:


X_train1, X_test1, y_train1, y_test1 = train_test_split(x, y, test_size=0.33)


print("proportion of classes:\n", y_test1.value_counts()/len(y_test1))
###by spliting randomly we almost have the same proportion of classa 0 and 1


# In[34]:


#### SANDARDIZE THE DATA ########
#preprocessing.StandardScaler().fit(X_train)
X_train1 = preprocessing.StandardScaler().fit_transform(X_train1)
X_test1 = preprocessing.StandardScaler().fit_transform(X_test1)


# In[35]:


pca = PCA(.95)
pca.fit(X_train1)
X_train1 = pca.transform(X_train1)
X_test1 = pca.transform(X_test1)
pca.n_components_


# ### we start apllying machine learning algorithme and we try to get the best model without overfitting

# ### the first algoritms are presents here and we will try to improve accuracy

# In[36]:


len(X_train)
# Train
clf = DecisionTreeClassifier()
clf.fit(X_train,y_train)
y_pred = clf.predict(X_test)
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))


# In[37]:


clf = DecisionTreeClassifier(criterion='gini', splitter='random', min_samples_split=20)
clf.fit(X_train1,y_train1)
y_predict1 = clf.predict(X_test1)
print("Accuracy :",metrics.accuracy_score(y_test1, y_predict1))
confusion_matrix(y_test1, y_predict1)

n = len(y_test1)



TP = []
TN = []
FP = []
FN = []

#initialize

if ((y_predict1[0]==0) and (y_test1.iloc[0]==0)):
    TP.append((1))
    TN.append(0)
    FP.append(0)
    FN.append(0)
if ((y_predict1[0]==0) and (y_test1.iloc[0]==1)):
    TP.append(0)
    TN.append(0)
    FP.append(0)
    FN.append(1)
if ((y_predict1[0]==1) and (y_test1.iloc[0]==0)):
    TP.append(0)
    TN.append(0)
    FP.append(1)
    FN.append(0)
if ((y_predict1[0]==1) and (y_test1.iloc[0]==1)):
    TP.append(0)
    TN.append(1)
    FP.append(0)
    FN.append(0)
for k in range(1,n):
    if ((y_predict1[k]==0) and (y_test1.iloc[k]==0)):
        TP.append((TP[k-1]+1))
        TN.append(TN[k-1]+0)
        FP.append(FP[k-1]+0)
        FN.append(FN[k-1]+0)
        
    if ((y_predict1[k]==0) and (y_test1.iloc[k]==1)):
        TP.append((TP[k-1]+0))
        TN.append(TN[k-1]+0)
        FP.append(FP[k-1]+0)
        FN.append(FN[k-1]+1)
    if ((y_predict1[k]==1) and (y_test1.iloc[k]==0)):
        TP.append((TP[k-1]+0))
        TN.append(TN[k-1]+0)
        FP.append(FP[k-1]+1)
        FN.append(FN[k-1]+0)
    if ((y_predict1[k]==1) and (y_test1.iloc[k]==1)):
        TP.append((TP[k-1]+0))
        TN.append(TN[k-1]+1)
        FP.append(FP[k-1]+0)
        FN.append(FN[k-1]+0)
        
TP = np.asarray(TP)
TN = np.asarray(TN)
FP = np.asarray(FP)
FN = np.asarray(FN)

TP = TP/TP[-1]
FP= FP/FP[-1]
splt.figure()

splt.plot(FP,TP)


# In[ ]:


#plot_tree(clf.fit(x,y)) 

# this clearly show that we need to focus on the feature selection.


# In[ ]:


confusion_ma = confusion_matrix(y_test, y_pred)

TP = confusion_ma[1, 1]
TN = confusion_ma[0, 0]
FP = confusion_ma[0, 1]
FN = confusion_ma[1, 0]

#Sensitivity
print('Sensitivity', TP / float(FN + TP))

#Specificity
print('Specificity', TN / (TN + FP))
#########################################!!! LOW SENSITIVITY!!!


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.33)
gnb = GaussianNB()
gnb.fit(X_train, y_train)
y_pred = gnb.predict(X_test)

print("Accuracy:",metrics.accuracy_score(y_test, y_pred))


# In[ ]:


confusion_ma = confusion_matrix(y_test, y_pred)

TP = confusion_ma[1, 1]
TN = confusion_ma[0, 0]
FP = confusion_ma[0, 1]
FN = confusion_ma[1, 0]

#Sensitivity
print('Sensitivity', TP / float(FN + TP))

#Specificity
print('Specificity', TN / (TN + FP))


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.33)
neigh = KNeighborsClassifier(n_neighbors=5)
neigh.fit(X_train, y_train)
y_pred = neigh.predict(X_test)

print("Accuracy:",metrics.accuracy_score(y_test, y_pred))


# In[ ]:


confusion_ma = confusion_matrix(y_test, y_pred)

TP = confusion_ma[1, 1]
TN = confusion_ma[0, 0]
FP = confusion_ma[0, 1]
FN = confusion_ma[1, 0]

#Sensitivity
print('Sensitivity', TP / float(FN + TP))

#Specificity
print('Specificity', TN / (TN + FP))


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.33)

clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(15,), random_state=1)
clf.fit(X_train,y_train)

y_pred = gnb.predict(X_test)
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))


# In[ ]:


confusion_ma = confusion_matrix(y_test, y_pred)

TP = confusion_ma[1, 1]
TN = confusion_ma[0, 0]
FP = confusion_ma[0, 1]
FN = confusion_ma[1, 0]

#Sensitivity
print('Sensitivity', TP / float(FN + TP))

#Specificity
print('Specificity', TN / (TN + FP))


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.33)


clf = svm.SVC()
clf.fit(X_train,y_train)

y_pred = clf.predict(X_test)
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))


# ###  we will go more in detail to have a better model

# In[ ]:


### Kneighboor classifier

neigh = KNeighborsClassifier(n_neighbors=3, weights='uniform', algorithm='auto', p=2)
neigh.fit(X_train, y_train)

y_predict = neigh.predict(X_test)

print("Accuracy:",metrics.accuracy_score(y_test, y_pred))

for k in range(1,8):
        
    neigh = KNeighborsClassifier(n_neighbors=k, weights='uniform', algorithm='auto', p=2)
    neigh.fit(X_train, y_train)

    y_predict = neigh.predict(X_test)

    print("Accuracy with ",k, "neighbors " ,metrics.accuracy_score(y_test, y_predict))

### there is no need to change the value of k


for k in range(1,8):
        
    neigh = KNeighborsClassifier(n_neighbors=k, weights='distance', algorithm='auto', p=2)
    neigh.fit(X_train, y_train)

    y_predict = neigh.predict(X_test)

    print("Accuracy with ",k, "neighbors and weight distance " ,metrics.accuracy_score(y_test, y_predict))

### there is no need to change the weight.

###The best value is made for 6 neighbors and uniform weigth


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




