import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#import dataset
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

bfriday_data = pd.read_csv("C:/Users/BHIWANDKAR/PycharmProjects/SONALIProject/BlackFriday.csv")

#print head
print(bfriday_data.head(10))
#check shape of dataset
print(bfriday_data.shape)

print(bfriday_data.info())
#missing values
print(bfriday_data.isnull().sum())

#visualization
plt.figure(figsize=(8,5))
sns.countplot(data=bfriday_data, x='Gender', palette='mako')
plt.show()

#marital status
plt.figure(figsize=(8,5))
sns.barplot(data=bfriday_data, x='Gender', y='Marital_Status')
plt.show()

#purchase status
plt.figure(figsize=(8,5))
sns.barplot(data=bfriday_data, x='Gender', y='Purchase')
plt.show()

#occupation status
plt.figure(figsize=(8,5))
sns.barplot(data=bfriday_data, x='Occupation', y='Purchase') #occupation has direct impact on purchase
plt.show()

#comparing male and female genders with hue
plt.figure(figsize=(8,5))
sns.barplot(data=bfriday_data, x='Occupation', y='Purchase', hue='Gender')
plt.show()

#outlier detection
#checking presence of ouutlier
plt.figure(figsize=(8,5))
sns.boxplot(data=bfriday_data, x='Gender', y='Purchase')
plt.show()

#occupation outlier
plt.figure(figsize=(8,5))
sns.boxplot(data=bfriday_data, x='Occupation', y='Purchase')
plt.show()

#purchase outlier
plt.figure(figsize=(8,5))
sns.boxplot(data=bfriday_data, x='Age', y='Purchase')
plt.show()

#product category outlier
plt.figure(figsize=(8,5))
sns.boxplot(data=bfriday_data, x='Product_Category_1', y='Purchase')
plt.show()

#Data prepossing
print(bfriday_data['Product_Category_2'].fillna(bfriday_data['Product_Category_2'].mean(),inplace=True))
print(bfriday_data['Product_Category_1'].mode())

#only want 5 value and not 0
print(bfriday_data['Product_Category_1'].mode()[0])

bfriday_data = bfriday_data.drop('Product_Category_2', axis=1)
#encode categorical column
print(bfriday_data['Product_ID'].value_counts())
print(bfriday_data['Product_Category_1'].value_counts())

# Select relevant features
features = ['Product_ID', 'Occupation', 'Age', 'Marital_Status']
bfriday_data.drop('Age', axis=1, inplace=True)
from sklearn.preprocessing import LabelEncoder

label_encoder = LabelEncoder()
bfriday_data['Product_ID'] = label_encoder.fit_transform(bfriday_data['Product_ID'])
bfriday_data['Gender'] = label_encoder.fit_transform(bfriday_data['Gender'])
bfriday_data['City_Category'] = label_encoder.fit_transform(bfriday_data['City_Category'])
bfriday_data['Stay_In_Current_City_Years'] = label_encoder.fit_transform(bfriday_data['Stay_In_Current_City_Years'])

print(bfriday_data)


#Correlation betn all the column and quality column
correlation=bfriday_data.corr()

#Positive Correlation
#Negative Correlation
plt.figure(figsize=(10,10))
sns.heatmap(correlation,cbar=True,square=True,fmt='.1f',annot=True,annot_kws={'size': 8},cmap='Blues')
plt.show()

#separate the data and label
X= bfriday_data.drop('Purchase',axis=1)
print(X)

#label Binarization
Y = bfriday_data['Purchase'].apply(lambda y_value: 1 if y_value>=7 else 0)
print(Y)

#Train & Test Split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=3)
print(Y.shape,Y_train.shape,Y_test.shape)

from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
pipeline = make_pipeline(SimpleImputer(strategy='mean'), LinearRegression())
pipeline.fit(X_train, y_train)
y_pred = pipeline.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error:", mse)


