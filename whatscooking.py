import numpy as np 
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

#import dataset
recipe = pd.read_json('C:/Users/BHIWANDKAR/PycharmProjects/SONALIProject/train.json')

print(recipe.head())
print('Shape:',recipe.shape)
print('Columns:',recipe.columns)
print('Whether Null exists:\n',recipe.isnull().sum())

print(recipe['cuisine'].nunique())
print(recipe['cuisine'].unique())

print(recipe['ingredients'][0])
print(recipe['ingredients'][6])

#To many entries are there for each cuisine
print(recipe['cuisine'].value_counts())

# bar plot for count of entries for each cuisine
x = recipe['cuisine'].value_counts().index
y = recipe['cuisine'].value_counts().values

df = pd.DataFrame({'Cuisine':x,'These many entries':y})

#fig = sns.countplot(recipe['cuisine'])
plt.figure(figsize=(8,5))
sns.barplot(data=df, x='Cuisine', y='These many entries')
plt.show()

from wordcloud import WordCloud
x= recipe['cuisine'].values

plt.subplots(figsize = (8,8))

wordcloud = WordCloud (background_color = 'white',width = 712,height = 384,colormap = 'prism'    ).generate(' '.join(x))

plt.imshow(wordcloud) # image show
plt.axis('off') # to off the axis of x and y
plt.savefig('cuisines.png')
plt.show()

# list to store all ingredients
all_ingredients = []

for indiv_ingredient_list in recipe['ingredients'].values:
    for ingredient in indiv_ingredient_list:
        all_ingredients.append(ingredient)
print(len(all_ingredients)) # 4lacs ingredients

ingredients_together = pd.DataFrame(all_ingredients)
print(ingredients_together)
print(ingredients_together.value_counts()[0:30]) # for first 30

# bar plot for count of entries for each cuisine
x = ingredients_together.value_counts()[0:30].index.tolist()
y = ingredients_together.value_counts()[0:30].values

df = pd.DataFrame({'Ingredient':x,'These many entries':y})
#fig = sns.countplot(recipe['cuisine'])
plt.figure(figsize=(8,5))
sns.barplot(data=df, x='Ingredient',y='These many entries')
plt.show()

print(recipe['cuisine'].value_counts())
recipe['cuisine'] = recipe['cuisine'].map({'italian':1,
                       'mexican':2,
                       'southern_us':3,
                       'indian':4,
                       'chinese':5,
                       'french':6,
                       'cajun_creole':7,
                       'thai':8,
                       'japanese':9,
                       'greek':10,
                       'spanish':11,
                       'korean':12,
                       'vietnamese':13,
                       'moroccan':14,
                       'british':15,
                       'filipino':16,
                       'irish':17,
                       'jamaican':18,
                       'russian':19,
                       'brazilian':20
})

#Split input and output columns
X = recipe.iloc[:,-1]
y = recipe['cuisine']
print(X)
print(y)

#Train & Test Split
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=1)
X_train = pd.DataFrame(X_train)
X_test = pd.DataFrame(X_test)
print(X_train)

X_train['ingredients'] = X_train['ingredients'].apply(lambda x:  ' '.join(x))
X_test['ingredients'] = X_test['ingredients'].apply(lambda x:  ' '.join(x))
print(X_train)
print(X_test)

# Applying BoW
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer()
X_train_bow = cv.fit_transform(X_train['ingredients']).toarray()
X_test_bow = cv.transform(X_test['ingredients']).toarray()
print(X_train_bow.shape)

from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()

gnb.fit(X_train_bow,y_train)

y_pred = gnb.predict(X_test_bow)

from sklearn.metrics import accuracy_score,confusion_matrix
print('Acaccuracy_score = ',accuracy_score(y_test,y_pred))

