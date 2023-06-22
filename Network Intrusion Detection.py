import pandas as pd
import os
for dirname, filenames in os.walk('C:/nskdl'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split,StratifiedKFold,GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier,VotingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV,RandomizedSearchCV
from sklearn.svm import SVC
from sklearn import metrics
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import make_scorer, roc_auc_score
import scipy
from scipy import stats
import warnings
warnings.filterwarnings("ignore")
from sklearn.preprocessing import StandardScaler
import os
import warnings
import gc
from plotly.subplots import make_subplots
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import precision_recall_fscore_support, classification_report

# Configuration
warnings.simplefilter('ignore')
#pd.set_option('max_columns', 50)

columns = (['duration'
,'protocol_type'
,'service'
,'flag'
,'src_bytes'
,'dst_bytes'
,'land'
,'wrong_fragment'
,'urgent'
,'hot'
,'num_failed_logins'
,'logged_in'
,'num_compromised'
,'root_shell'
,'su_attempted'
,'num_root'
,'num_file_creations'
,'num_shells'
,'num_access_files'
,'num_outbound_cmds'
,'is_host_login'
,'is_guest_login'
,'count'
,'srv_count'
,'serror_rate'
,'srv_serror_rate'
,'rerror_rate'
,'srv_rerror_rate'
,'same_srv_rate'
,'diff_srv_rate'
,'srv_diff_host_rate'
,'dst_host_count'
,'dst_host_srv_count'
,'dst_host_same_srv_rate'
,'dst_host_diff_srv_rate'
,'dst_host_same_src_port_rate'
,'dst_host_srv_diff_host_rate'
,'dst_host_serror_rate'
,'dst_host_srv_serror_rate'
,'dst_host_rerror_rate'
,'dst_host_srv_rerror_rate'
,'attack'
,'level'])

df_train=pd.read_csv(r'C:\Users\BHIWANDKAR\PycharmProjects\SONALIProject\KDDTrain+.txt',header=None,names=columns)
df_test=pd.read_csv(r'C:\Users\BHIWANDKAR\PycharmProjects\SONALIProject\KDDTest+.txt',header=None,names=columns)

# def change_label(df):
#     df.attack.replace(['apache2','back','land','neptune','mailbomb','pod','processtable','smurf','teardrop','udpstorm','worm'],'Dos',inplace=True)
#     df.attack.replace(['ftp_write','guess_passwd','httptunnel','imap','multihop','named','phf','sendmail',
#        'snmpgetattack','snmpguess','spy','warezclient','warezmaster','xlock','xsnoop'],'R2L',inplace=True)
#     df.attack.replace(['ipsweep','mscan','nmap','portsweep','saint','satan'],'Probe',inplace=True)
#     df.attack.replace(['buffer_overflow','loadmodule','perl','ps','rootkit','sqlattack','xterm'],'U2R',inplace=True)

# change_label(df_train)
# change_label(df_test)

# with open("../input/kdd-cup-1999-data/training_attack_types",'r') as f:
#     print(f.read())

print(df_train)

print(df_train['attack'].unique())

print(df_train.hist(figsize=(16, 12)))

print(df_train.info())

df_train['attack'].describe()
print(df_train['attack'].value_counts())

print(df_train.duplicated().sum())
print(df_test.duplicated().sum())

print(df_train.isnull().sum())

print(df_train['attack'].unique())

print(df_train.head())

cat_features=[i for i in df_train.columns if df_train.dtypes[i]=='object']

print(cat_features)

print(df_train['protocol_type'].unique())

print(df_train['service'].value_counts())

print(df_train['flag'].unique())

print(df_train['attack'].value_counts())

# We will make the attack column binomial by classifying all attacks as one i.e. normal flow vs attack
df_train["binary_attack"]=df_train.attack.map(lambda a: "normal" if a == 'normal' else "abnormal")
df_train.drop('attack',axis=1,inplace=True)

df_test["binary_attack"]=df_test.attack.map(lambda a: "normal" if a == 'normal' else "abnormal")
df_test.drop('attack',axis=1,inplace=True)

print(df_train['binary_attack'].value_counts())

print(df_train.select_dtypes(['object']).columns)

from sklearn import preprocessing
le=preprocessing.LabelEncoder()
clm=['protocol_type', 'service', 'flag', 'binary_attack']
for x in clm:
    df_train[x]=le.fit_transform(df_train[x])
    df_test[x]=le.fit_transform(df_test[x])

print(df_train['service'].value_counts())

corr = df_train.corr()

plt.figure(figsize=(10,10))
sns.heatmap(corr)
plt.show()

df_train.drop('num_root',axis = 1,inplace = True)

#This variable is highly correlated with serror_rate and should be ignored for analysis.
#(Correlation = 0.9983615072725952)
df_train.drop('srv_serror_rate',axis = 1,inplace = True)

#This variable is highly correlated with rerror_rate and should be ignored for analysis.
#(Corredfation = 0.9947309539817937)
df_train.drop('srv_rerror_rate',axis = 1, inplace=True)

#This variable is highly correlated with srv_serror_rate and should be ignored for analysis.
#(Correlation = 0.9993041091850098)
df_train.drop('dst_host_srv_serror_rate',axis = 1, inplace=True)

#This variable is highly correlated with rerror_rate and should be ignored for analysis.
#(Correlation = 0.9869947924956001)
df_train.drop('dst_host_serror_rate',axis = 1, inplace=True)

#This variable is highly correlated with srv_rerror_rate and should be ignored for analysis.
#(Correlation = 0.9821663427308375)
df_train.drop('dst_host_rerror_rate',axis = 1, inplace=True)


#This variable is highly correlated with rerror_rate and should be ignored for analysis.
#(Correlation = 0.9851995540751249)


df_test.drop('num_root',axis = 1,inplace = True)

#This variable is highly correlated with serror_rate and should be ignored for analysis.
#(Correlation = 0.9983615072725952)
df_test.drop('srv_serror_rate',axis = 1,inplace = True)

#This variable is highly correlated with rerror_rate and should be ignored for analysis.
#(Corredfation = 0.9947309539817937)
df_test.drop('srv_rerror_rate',axis = 1, inplace=True)

#This variable is highly correlated with srv_serror_rate and should be ignored for analysis.
#(Correlation = 0.9993041091850098)
df_test.drop('dst_host_srv_serror_rate',axis = 1, inplace=True)

#This variable is highly correlated with rerror_rate and should be ignored for analysis.
#(Correlation = 0.9869947924956001)
df_test.drop('dst_host_serror_rate',axis = 1, inplace=True)

#This variable is highly correlated with srv_rerror_rate and should be ignored for analysis.
#(Correlation = 0.9821663427308375)
df_test.drop('dst_host_rerror_rate',axis = 1, inplace=True)

#This variable is highly correlated with rerror_rate and should be ignored for analysis.
#(Correlation = 0.9851995540751249)

x_train=df_train.drop('binary_attack',axis=1)
y_train=df_train["binary_attack"]

x_test=df_test.drop('binary_attack',axis=1)
y_test=df_test["binary_attack"]

# from sklearn.feature_selection import mutual_info_classif
# mutual_info = mutual_info_classif(x_train, y_train)
# mutual_info = pd.Series(mutual_info)
# mutual_info.index = x_train.columns
# mutual_info.sort_values(ascending=False)

# model for the binary classification
binary_model = RandomForestClassifier()
binary_model.fit(x_train, y_train)
binary_predictions = binary_model.predict(x_test)

# calculate and display our base accuracy
base_rf_score = accuracy_score(binary_predictions,y_test)
print(base_rf_score)

# mutual_info.sort_values(ascending=False).plot.bar(figsize=(20, 8));

# from sklearn.feature_selection import SelectKBest
# sel_five_cols = SelectKBest(mutual_info_classif, k=20)
# sel_five_cols.fit(x_train, y_train)
# x_train.columns[sel_five_cols.get_support()]

# col=['service', 'flag', 'src_bytes', 'dst_bytes', 'logged_in',
#        'same_srv_rate', 'diff_srv_rate', 'dst_host_srv_count',
#        'dst_host_same_srv_rate', 'dst_host_diff_srv_rate']
# x_train=x_train[col]
# x_test=x_test[col]

plt.figure(figsize=(10,10))
p=sns.heatmap(x_train.corr(), annot=True,cmap ='RdYlGn')

x_train['flag'].value_counts().plot(kind='bar')

print(x_train)

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
x_train= scaler.fit_transform(x_train)
x_test= scaler.fit_transform(x_test)

# model for the binary classification
binary_model = RandomForestClassifier()
binary_model.fit(x_train, y_train)
binary_predictions = binary_model.predict(x_test)

# calculate and display our base accuracty
base_rf_score = accuracy_score(binary_predictions,y_test)
print('base_rf_score = ',base_rf_score)

print('x_train =\n',x_train)

models = {}
# Logistic Regression
from sklearn.linear_model import LogisticRegression
models['Logistic Regression'] = LogisticRegression(multi_class='multinomial')

# Support Vector Machines
from sklearn.svm import LinearSVC
models['Support Vector Machines linear'] = LinearSVC()
models['Support Vector Machines plonomial'] = SVC(kernel='poly')
models['Support Vector Machines RBf'] =  SVC(C=100.0)

# Decision Trees
from sklearn.tree import DecisionTreeClassifier
models['Decision Trees'] = DecisionTreeClassifier(max_depth=3)

# Random Forest
from sklearn.ensemble import RandomForestClassifier
models['Random Forest'] = RandomForestClassifier()

# Naive Bayes
from sklearn.naive_bayes import GaussianNB
models['Naive Bayes'] = GaussianNB()

# K-Nearest Neighbors
from sklearn.neighbors import KNeighborsClassifier
models['K-Nearest Neighbor'] = KNeighborsClassifier(n_neighbors=20)

from sklearn.metrics import accuracy_score, precision_score, recall_score

accuracy, precision, recall = {}, {}, {}

for key in models.keys():

    # Fit the classifier
    models[key].fit(x_train, y_train)

    # Make predictions
    predictions = models[key].predict(x_test)

    # Calculate metrics
    accuracy[key] = accuracy_score(predictions, y_test)
    precision[key] = precision_score(predictions, y_test)
    recall[key] = recall_score(predictions, y_test)

df_model = pd.DataFrame(index=models.keys(), columns=['Accuracy', 'Precision', 'Recall'])
df_model['Accuracy'] = accuracy.values()
df_model['Precision'] = precision.values()
df_model['Recall'] = recall.values()

print('df_model =\n',df_model)

def fun():
    model = Sequential()

    #here 30 is output dimension
    model.add(Dense(36,input_dim =36,activation = 'relu',kernel_initializer='random_uniform'))

    #in next layer we do not specify the input_dim as the model is sequential so output of previous layer is input to next layer
    model.add(Dense(1,activation='sigmoid',kernel_initializer='random_uniform'))

    #5 classes-normal,dos,probe,r2l,u2r
    model.add(Dense(2,activation='softmax'))

    #loss is categorical_crossentropy which specifies that we have multiple classes

    model.compile(loss ='categorical_crossentropy',optimizer = 'adam',metrics = ['accuracy'])

    return model

from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
model7 = KerasClassifier(build_fn=fun,epochs=100,batch_size=64)

model7.fit(x_train, y_train.values.ravel())
