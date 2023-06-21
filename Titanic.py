import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

titanic_data = pd.read_csv("C:/Users/admin/PycharmProjects/SONALIProject/venv/train (1).csv")

titanic_data.head(10)
#more info
titanic_data.info()
#missing values
titanic_data.isnull().sum()

#dop
titanic_data = titanic_data.drop(columns='Cabin', axis=1)

#replace missing value in age column with mean value
titanic_data['Age'].fillna(titanic_data['Age'].mean(),inplace=True)
print(titanic_data['Embarked'].mode())

#only want s value and not 0
print(titanic_data['Embarked'].mode()[0])
print(titanic_data)

#replace missing value in embarked column with mode value
titanic_data['Embarked'].fillna(titanic_data['Embarked'].mode(),inplace=True)
titanic_data.isnull().sum()
print(titanic_data)

#stastical operations
titanic_data.describe()
print(titanic_data)

#survived number
print(titanic_data['Survived'].value_counts())

#data part
sns.set()#gives theme for plot
#count how many survived
sns.countplot(data=titanic_data, x='Survived')
#0 indicates persons who dies and 1 indicates who survived
plt.xlabel("Survived")
plt.ylabel("Count")
plt.show()

#check count of male and female
print(titanic_data['Sex'].value_counts())

#plot graph of sex column
sns.countplot(data=titanic_data, x='Sex', color='green')
plt.xlabel("Sex")
plt.ylabel("Count")
plt.show()

#check number of survival geder wise
sns.countplot(data=titanic_data, x='Sex',hue='Survived')
#with hue we can see how many males survived and how many females survived and died
plt.show()

#for p class
sns.set()
sns.countplot(data=titanic_data,x='Pclass')
plt.show()

#to miss person from 1st class who survived and person from 2 class
sns.countplot(data=titanic_data, x='Pclass',hue='Survived')
plt.show()

#encode categorical column
print(titanic_data['Sex'].value_counts())
print(titanic_data['Embarked'].value_counts())

# Select relevant features
features = ['Pclass', 'Sex', 'Age', 'Fare']

# Handle missing values
titanic_data = titanic_data[features + ['Survived']].dropna()
# Convert categorical variables into numerical representations
titanic_data['Sex'] = titanic_data['Sex'].map({'male': 0, 'female': 1})
# Split the data into training and testing sets
X = titanic_data[features]
y = titanic_data['Survived']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
 # Initialize the Random Forest Classifier
rf_classifier = RandomForestClassifier()

# Train the classifier
rf_classifier.fit(X_train, y_train)
# Make predictions
y_pred = rf_classifier.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)