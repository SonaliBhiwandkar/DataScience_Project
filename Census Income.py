import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import mean_squared_error


df = pd.read_csv(r"C:/Users/BHIWANDKAR/PycharmProjects/SONALIProject/adult.csv")

#print head
print(df.head(10))
#check shape of dataset
print(df.shape)

print(df.info())
#missing values
print(df.isnull().sum())

#visualization
plt.figure(figsize=(8,5))
sns.countplot(data=df, x='workclass', palette='mako')
plt.show()

#educationandpayment
plt.figure(figsize=(8,5))
sns.barplot(data=df, x='education', y='hours.per.week')
plt.show()

#race and sex
plt.figure(figsize=(8,5))
sns.barplot(data= df, x='race', y='age')
plt.show()

#occupation and capital loss
plt.figure(figsize=(8,5))
sns.barplot(data=df, x='occupation', y='capital.loss') #occupation has direct impact on purchase
plt.show()

rf = RandomForestClassifier(random_state=1)
lr = LogisticRegression(random_state=1, max_iter=1000)
dt = DecisionTreeClassifier(random_state=0)
gbm = GradientBoostingClassifier(n_estimators=10)

print(df.isnull().sum())

le = LabelEncoder()
df['workclass'] = le.fit_transform(df['workclass'])
df['education'] = le.fit_transform(df['education'])
df['marital.status'] = le.fit_transform(df['marital.status'])
df['occupation'] = le.fit_transform(df['occupation'])
df['relationship'] = le.fit_transform(df['relationship'])
df['race'] = le.fit_transform(df['race'])
df['sex'] = le.fit_transform(df['sex'])
df['native.country'] = le.fit_transform(df['native.country'])
df['income'] = le.fit_transform(df['income'])

x = df.drop('income', axis=1)
y = df['income']

from sklearn.preprocessing import StandardScaler

# Select numerical features for scaling
numerical_features = ['age', 'fnlwgt', 'education.num', 'capital.gain', 'capital.loss', 'hours.per.week']

# Scale numerical features
scaler = StandardScaler()
df[numerical_features] = scaler.fit_transform(df[numerical_features])


x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=0, test_size=0.2)

# Logistic regression

lr.fit(x_train, y_train)
y_pred1 = lr.predict(x_test)
print("Accuracy by logistic regression:",accuracy_score(y_test, y_pred1))

# Random forest

rf.fit(x_train, y_train)
y_pred2 = rf.predict(x_test)
print("Accuracy by Random forest:",accuracy_score(y_test, y_pred2))

# Gradient Boosting

gbm = GradientBoostingClassifier(n_estimators=300,
                                 learning_rate=0.05,
                                 random_state=100,
                                 max_features=5)
gbm.fit(x_train, y_train)
y_pred3 = gbm.predict(x_test)
print("Accuracy by GBM:",accuracy_score(y_test, y_pred3))
