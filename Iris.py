import pandas as pd
import matplotlib as plt
import  seaborn as sns

df=pd.read_csv("C:/Users/BHIWANDKAR/PycharmProjects/SONALIProject/Iris.csv")

#preparing  X  and Y
x=df.drop('Id',axis=1)
x=x.drop('Species',axis=1)
y=df['Species']
print(x)
print(y)

from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

bestfeatures = SelectKBest(score_func=chi2,k='all')
fit=bestfeatures.fit(x,y)
dfscores=pd.DataFrame(fit.scores_)
dfcolumns=pd.DataFrame(x.columns)
featuresScores=pd.DataFrame(x.columns)
featuresScores = pd.concat([dfcolumns,dfscores],axis=1)
featuresScores.columns= ['Specs','Score']

print(featuresScores)
print(bestfeatures)

print(df)
print(df.shape)
print(df.describe())
print(df.isna().sum())
print(df.describe())
print(df.head())
print(df.head(150))
print(df.tail(100))

n = len(df[df['Species'] == 'Iris-versicolor'])
print("No of Versicolor in Dataset:",n)

n1 = len(df[df['Species'] == 'Iris-virginica'])
print("No of Virginica in Dataset:",n1)

from sklearn.ensemble import ExtraTreesClassifier
import matplotlib.pyplot as plt

fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
ax.axis('equal')
l = ['Iris-versicolor', 'Iris-setosa', 'Iris-virginica']
s = [50,50,50]
ax.pie(s, labels = l,autopct='%1.2f%%')
plt.show()

model = ExtraTreesClassifier()
model.fit(x,y)
print(model.feature_importances_)

feat_importance=pd.Series(model.feature_importances_,index=x.columns)
feat_importance.nlargest(4).plot(kind='barh')
plt.show()


#identifying Outliners by ploting
from matplotlib import pyplot as plt
import seaborn as sns

plt.figure(1)
plt.boxplot([df['SepalLengthCm']])
plt.figure(2)
plt.boxplot([df['SepalWidthCm']])
plt.show()

sns.boxplot(df['SepalLengthCm'])
plt.show()

df.plot(kind ='density',subplots = True, layout =(3,3),sharex = False)
plt.show()

X = df['SepalLengthCm'].values.reshape(-1,1)
print(X)
Y = df['SepalWidthCm'].values.reshape(-1,1)
print(X)

plt.xlabel("Sepal Length")
plt.ylabel("Sepal Width")
plt.scatter(X,Y,color='b')
plt.show()

#Dealing With Outliers using Interquantile Range
print(df['SepalLengthCm'])
Q1=df['SepalLengthCm'].quantile(0.25)
Q3=df['SepalLengthCm'].quantile(0.75)

IQR=Q3 - Q1
print(IQR)

upper=Q3+1.5*IQR
lower=Q1-1.5*IQR

print(upper)
print(lower)

out1=df[df['SepalLengthCm']< lower].values
out2=df[df['SepalLengthCm']>upper].values

print(df['SepalLengthCm'].replace(out1,lower,inplace=True))
print(df['SepalLengthCm'].replace(out2,upper,inplace=True))
print(df['SepalLengthCm'])

#Numerial to Catagorial
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import  LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
rf=RandomForestClassifier()


df['SepalLengthCm']=pd.cut(df['SepalLengthCm'],3,labels=['0','1','2'])
df['SepalWidthCm']=pd.cut(df['SepalWidthCm'],3,labels=['0','1','2'])
df['PetalLengthCm']=pd.cut(df['PetalLengthCm'],3,labels=['0','1','2'])
df['PetalWidthCm']=pd.cut(df['PetalWidthCm'],3,labels=['0','1','2'])

print(df)


#Dealing with missing values
'''
1. Use Drop (df.drop())
2.use Replace (df.replace("black","DOS"))
3.Fill NA ()
df['Item_Weight'].fillna((df['Item_Weight'].mean()/.median()/.mode()),implace=True)

df['Item_Size'].fillna(('Medium'),implace=True) 
'''


from collections import Counter
print(Counter(y))
from imblearn.over_sampling import RandomOverSampler
ros=RandomOverSampler(random_state=0)
x,y =ros.fit_resample(x,y)
print(Counter(y))


#principal complement Analysis

from  sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

logr=LogisticRegression()
pca=PCA(n_components=2)

X=df.drop('Id',axis=1)
X=X.drop('Species',axis=1)
Y=df['Species']

pca.fit(X)
X=pca.transform(X)

print(X)

X_train,X_test,y_train,y_test=train_test_split(X,Y,random_state=0,test_size=0.2)
logr.fit(X_train,y_train)

y_pred=logr.predict(X_test)
print('Accuracy_score = ',accuracy_score(y_test,y_pred))