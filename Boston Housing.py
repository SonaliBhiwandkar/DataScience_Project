import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

boston_data = pd.read_csv("C:/Users/admin/PycharmProjects/SONALIProject/venv/BostonHousing1.csv")

print(boston_data.keys())
print(boston_data.head(10))
print(boston_data.info())
print(boston_data.isnull().sum())

print(boston_data._data)

#dop
boston_data = boston_data.drop(columns='zn', axis=1)
print(boston_data.drop)

#replace missing value in age column with mean value
boston_data['age'].fillna(boston_data['age'].mean(),inplace=True)
print(boston_data['rad'].mode())

#only want 24 value and not 0
print(boston_data['rad'].mode()[0])

#replace missing value in embarked column with mode value
print(boston_data['rad'].fillna(boston_data['rad'].mode(),inplace=True))
print(boston_data.isnull().sum())

#stastical operations
print(boston_data.describe())

print(boston_data.value_counts())
boston_data = boston_data.drop(columns='age', axis=1)
print(boston_data.drop)

#plot graph
plt.figure(figsize=(12,6))
plt.hist(boston_data['crim'], color='g');
plt.xlabel('Crime')
plt.ylabel('Frequency')
plt.show()

sns.countplot(data=boston_data, x='tax')
plt.show()

sns.histplot(boston_data['rad'], color='red')
plt.show()

sns.distplot(boston_data['rad'], color='red')
plt.show()

sns.countplot(data=boston_data,x='medv',palette='cubehelix')
plt.show()

sns.distplot(boston_data['medv'])
plt.show()

print(boston_data.corr())

sns.jointplot(x='rad', y='medv', data=boston_data,kind='hex',color='g')
plt.show()

sns.jointplot(x='ptratio',y='medv',data=boston_data,kind='reg',color='r')
plt.show()

sns.boxplot(data=boston_data, x='medv')
plt.show()

#encode categorical column
print(boston_data['age'].value_counts())
print(boston_data['rad'].value_counts())

#training model
X=boston_data.drop('medv', axis=1)
y= boston_data['medv']
import sklearn
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test=train_test_split(X,y,test_size=0.33,random_state=5)
print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)

from sklearn.linear_model import LinearRegression
lin_reg= LinearRegression()
lin_reg.fit(X_train, y_train)

predictions= lin_reg.predict(X_test)
plt.scatter(y_test, predictions)
plt.xlabel('Prices')
plt.ylabel('Predicted Prices')
plt.title('Prices vs Predicted prices')
plt.show()

lin_reg.score(X_test, y_test)
error= y_test-predictions
sns.distplot(error)
plt.show()
#accuracy
accuracy_score= sklearn.metrics.mean_squared_error(y_test, predictions)
print("Accuracy:",accuracy_score)