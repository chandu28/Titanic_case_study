# -*- coding: utf-8 -*-
"""
Created on Wed Dec 12 05:11:09 2018

@author: CHANDU
"""

import os
import numpy as np
import pandas as pd
from scipy.stats import skew,mode
import matplotlib.pyplot as plt
import seaborn as sns


os.chdir('C:\\Users\\Chandu\\Desktop\\IMAR DATA\\titanic data')
ti_train= pd.read_csv('train.csv')
ti_test= pd.read_csv('test.csv')

#------------------------------labels----------------------------
ti_train.columns
#'PassengerId', 'Survived', 'Pclass', 'Name', 'Sex', 'Age', 'SibSp',
#       'Parch', 'Ticket', 'Fare', 'Cabin', 'Embarked']

#---------------------------------exploratory analytsis
#------------------null values using seaborn------------------------
#--------------------------yticklabels if  false will not plot columns names which have all false values--------

sns.heatmap(ti_train.isnull(),yticklabels=False,cbar=False,cmap='viridis')

def missing(x):
    return sum(x.isnull())

ti_train.apply(missing, axis=0)

#------------------------------explpratrory analysisi--------------
#-----------------count plots-----------------------------------
sns.set_style('whitegrid')
sns.countplot(x='Survived',data=ti_train)


#-----------------based on sex------------------------------
sns.countplot(x='Survived',hue='Sex',palette='RdBu_r',data=ti_train)
sns.countplot(x='Survived',hue='Pclass',data=ti_train)

#--------------------------------------kde gaussioan,hist------------------
sns.distplot(ti_train['Age'].dropna(),kde=False,bins=50)
sns.distplot(ti_train['Age'].dropna(),hist=True, kde=True,bins=30)

sns.countplot(x='SibSp',data=ti_train)

#--------------------hist by using normal plt

ti_train['Fare'].hist()


#---------------------missing values based on  other features
#heat map
sns.heatmap(ti_train.isnull(),yticklabels=False,cbar=False,cmap='viridis')
#box plot
plt.figure(figsize=(10,4))
sns.boxplot(x= 'Pclass',y='Age',data= ti_train)
ti_train[ti_train['Pclass']==1]['Age'].mean()
ti_train[ti_train['Pclass']==2]['Age'].mean()
ti_train[ti_train['Pclass']==3]['Age'].mean()

def impute_age(col):
    Age= col[0]
    Pclass= col[1]
    
    if pd.isnull(Age):
        
        if Pclass==1:
            return 38
        elif Pclass==2:
            return 30
        else:
            return 25
    else:
        return Age
    
    
ti_train['Age']=ti_train[['Age','Pclass']].apply(impute_age, axis=1)

ti_train.drop('Cabin',axis=1,inplace=True)


#-------------------using get_dummies for feature variable

age= pd.get_dummies(ti_train['Sex'],drop_first= True)

#drop first to drop firdt column to avoid confusion
#--------------stored males as 1 and 0 as female

embarked= pd.get_dummies(ti_train['Embarked'],drop_first=True)


ti_train= pd.concat([ti_train,age,embarked],axis=1)


#-----------------------removing unwanted variable---------------------

#try adding dummies for pclass after fitting model---------------
ti_train.drop(['Sex','Embarked','Ticket','Name'],axis=1,inplace=True)
ti_train.drop(['PassengerId'],axis=1,inplace=True)#unique record


#--------train test split--------------

X= ti_train.iloc[:,1:]
Y= ti_train.iloc[:,0:1]


from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

x_train,x_test,y_train,y_test= train_test_split(X,Y,test_size=0.2)

from sklearn import linear_model
lm= linear_model.LogisticRegression()

lm.fit(x_train,y_train)
predicted_lm_train= lm.predict(x_train)
#evaluation
confusion_matrix_x_train= confusion_matrix(y_train,predicted_lm_train)
accuracy_matrix_x_train= accuracy_score(y_train,predicted_lm_train)
print(accuracy_matrix_x_train)

#---------------------classification report
from sklearn.metrics import classification_report
print(classification_report(y_train,predicted_lm_train))



