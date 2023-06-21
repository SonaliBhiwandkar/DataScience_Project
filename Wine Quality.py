import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

wine_dataset = pd.read_csv("C:/Users/admin/Downloads/winequality-red.csv")

print(wine_dataset.shape)
#first Data
print(wine_dataset.head())
#checking for missing values
print(wine_dataset.isnull().sum())

#Data Analysis and Visulizatio
print(wine_dataset.describe())  #stasticical measure of dataset

#number of values for each equality
sns.countplot(x='quality',data=wine_dataset)
plt.ylabel("Count")
plt.show()

#volatile acidity vs Quality
ploat= plt.figure(figsize=(5,5))
sns.barplot(x='quality',y='volatile acidity',data=wine_dataset)
plt.show()

#citric acid vs Quality
ploat= plt.figure(figsize=(5,5))
sns.barplot(x='quality',y='citric acid',data=wine_dataset)
plt.show()


#Correlation betn all the column and quality column
correlation=wine_dataset.corr()

#Positive Correlation
#Negative Correlation
plt.figure(figsize=(10,10))
sns.heatmap(correlation,cbar=True,square=True,fmt='.1f',annot=True,annot_kws={'size': 8},cmap='Blues')
plt.show()

#separate the data and label
X= wine_dataset.drop('quality',axis=1)
print(X)

#label Binarization
Y = wine_dataset['quality'].apply(lambda y_value: 1 if y_value>=7 else 0)
print(Y)

#Train & Test Split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=3)
print(Y.shape,Y_train.shape,Y_test.shape)



#Traning
model=RandomForestClassifier()
model.fit(X_train,Y_train)

#accuracy
X_test_prediction = model.predict(X_test)
test_data_accuracy = accuracy_score(X_test_prediction,Y_test)
print('Accuracy :',test_data_accuracy)


