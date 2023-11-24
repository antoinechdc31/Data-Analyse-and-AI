#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib
gend = pd.read_csv('gender_submission.csv')
gend.head(6)


# In[2]:


#Import the dataset
train = pd.read_csv('train.csv',sep=',')
train.head(2)


# In[3]:


#Import the dataset
test = pd.read_csv('test.csv')
test

sns.countplot(x='Survived', data=gend)
plt.title('Distribution des classes dans gend')
plt.show()
# In[4]:


train.head() #j'ai supprimé la colonne cabine dans l'excel train


# In[5]:


#I filled in the missing values of the age column in Excel Train
age_mean = train['Age'].mean()
train['Age'].fillna(age_mean, inplace=True)
train.head(10)


# In[6]:


#I filled in the missing values of the age column in Excel Test
age_mean = test['Age'].mean()
test['Age'].fillna(age_mean, inplace=True)
test.head(10)


# In[7]:


train['Embarked']=train['Embarked'].replace({'S':0,'C':1,'Q':2})
train.head(10)


# In[8]:


#I delete the columns not containing numerical values in the Excel Test
test=test.drop(['Name','Cabin','Ticket','PassengerId'],axis=1)
test['Sex'] = test['Sex'].replace({'female': 0, 'male': 1})
test['Embarked']=test['Embarked'].replace({'S':0,'C':1,'Q':2})
test.head()


# In[9]:


#I filled in the missing values of the 'Embarked' column in Excel Test and Train
emb_mean = test['Embarked'].mean()
test['Embarked'].fillna(emb_mean, inplace=True)
em_mean = train['Embarked'].mean()
train['Embarked'].fillna(em_mean, inplace=True)


# In[10]:


#I delete the columns not containing numerical values in the Excel train
train= train.drop('PassengerId', axis = 1)
train=train.drop('Ticket',axis=1)
train=train.drop('Name',axis=1)
train=train.drop('Cabin',axis=1)
train.head()


# In[11]:


#I associate numeric values with the column 'Sex'
train['Sex'] = train['Sex'].replace({'female': 0, 'male': 1})
train.head()


# In[12]:


#I delete rows with missing values in excel train
train.dropna()


# In[13]:


#I delete rows with missing values in excel test
test.dropna()


# In[14]:


X = train.drop('Survived', axis=1)
y = train['Survived']


# In[15]:


from sklearn.model_selection import train_test_split
# Split the dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[16]:


from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)


# In[17]:


train.info()


# In[18]:


y_pred = model.predict(X_test)

from sklearn.metrics import accuracy_score, classification_report

accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy}')

print('Classification Report:')
print(classification_report(y_test, y_pred))


# In[19]:


from sklearn.metrics import confusion_matrix

# Supposons que vous avez déjà créé et entraîné votre modèle RandomForest
# model = RandomForestClassifier(random_state=42)
# model.fit(X_train, y_train)

# Faire des prédictions sur l'ensemble de test
y_pred = model.predict(X_test)

# Calculer la matrice de confusion
conf_matrix = confusion_matrix(y_test, y_pred)

# Afficher la matrice de confusion
print("Matrice de Confusion:")
print(conf_matrix)


# In[20]:


# Use plot_confusion_matrix to display the confusion matrix
from sklearn.metrics import plot_confusion_matrix

plot_confusion_matrix(model, X_test, y_test, cmap='Blues', display_labels=['Not Survived', 'Survived'])
plt.title('Matrice de Confusion')
plt.show()


# In[21]:


from sklearn.preprocessing import LabelEncoder

# Get user input
user_pclass = int(input("Enter the person's passenger class (1, 2, or 3): "))
user_sex = input("Enter the person's sex (male/female): ")
user_age = float(input("Enter the person's age: "))
user_sibsp = int(input("Enter the number of siblings/spouses aboard: "))
user_parch = int(input("Enter the number of parents/children aboard: "))
user_fare = float(input("Enter the fare paid: "))
user_embarked = input("Enter the embarked port (C, Q, or S): ")

# Create a DataFrame with user input
user_data = pd.DataFrame({
    'Pclass': [user_pclass],
    'Sex': [user_sex],
    'Age': [user_age],
    'SibSp': [user_sibsp],
    'Parch': [user_parch],
    'Fare': [user_fare],
    'Embarked': [user_embarked]
})

# Encode categorical variables
le = LabelEncoder()
user_data['Sex'] = le.fit_transform(user_data['Sex'])
user_data['Embarked'] = le.fit_transform(user_data['Embarked'])

# Make predictions using the pre-trained model
prediction = model.predict(user_data)

# Display the result
if prediction[0] == 1:
    print("This person is predicted to survive.")
else:
    print("This person is predicted not to survive.")


# In[ ]:




