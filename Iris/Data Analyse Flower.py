#!/usr/bin/env python
# coding: utf-8

# In[1]:


#import all the librairy
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import GridSearchCV

iris = pd.read_csv('IRIS_ Flower_Dataset.csv')#import the dataset
iris.head()


# In[2]:


x = iris.drop('species', axis=1) #Extract features (X) and target variable (y)
y = iris['species']


# In[3]:


param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}


# In[4]:


# Split the dataset into training and test sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=12)


# In[5]:


# Create an initial RandomForestClassifier model
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=12)
rf_classifier.fit(x_train, y_train)# Train the initial model
rf_predictions = rf_classifier.predict(x_test)


# In[6]:


#Evaluate the Random Forest classifier
rf_accuracy = accuracy_score(y_test, rf_predictions)
print("\nRandom Forest Accuracy:", rf_accuracy)
print("Confusion Matrix:\n", confusion_matrix(y_test, rf_predictions))
print("Classification Report:\n", classification_report(y_test, rf_predictions))


# In[7]:



# Make predictions on the test set with the modified model
tuned_predictions = rf_classifier_tuned.predict(x_test)

# Measure the performance of the modified model
tuned_accuracy = accuracy_score(y_test, tuned_predictions)

# Make predictions on the test set with the initial model
initial_predictions = rf_classifier.predict(x_test)

# Measure initial performance
initial_accuracy = accuracy_score(y_test, initial_predictions)

# Calculate accuracy improvement
accuracy_improvement = tuned_accuracy - initial_accuracy

# Bar chart plot
improvement_data = [initial_accuracy, tuned_accuracy]
labels = ['Initial Model', 'Tuned Model']
colors = ['darkorange', 'darkgreen']

plt.bar(labels, improvement_data, color=colors)
plt.ylabel('Accuracy')
plt.title('Accuracy Improvement')
plt.show()


# In[8]:


feature_importances = rf_classifier_tuned.feature_importances_

# Extract feature names (columns)
feature_names = x.columns

# Sort feature indices in descending order of importance
indices = np.argsort(feature_importances)[::-1]

# Draw the graph of the importance of the characteristics
plt.figure(figsize=(10, 6))
plt.bar(range(len(feature_importances)), feature_importances[indices], align='center')
plt.xticks(range(len(feature_importances)), feature_names[indices], rotation=45)
plt.xlabel('Feature')
plt.ylabel('Importance')
plt.title('Feature Importances in Random Forest Model')
plt.show()


# In[9]:


from sklearn.metrics import confusion_matrix
import seaborn as sns

class_names = list(set(y)) # Get unique class names

# Calculate the confusion matrix
cm = confusion_matrix(y_test, tuned_predictions)

# Plot the confusion matrix with Seaborn
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='g', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()


# In[ ]:





# In[ ]:




