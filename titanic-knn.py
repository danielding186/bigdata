# An exmaple from big-data-class using kNN and the accurary is 0.565
# Data from https://www.kaggle.com/c/titanic
# coding: utf-8

# In[11]:

import pandas as pd
from pandas.api.types import is_string_dtype,is_numeric_dtype
from sklearn import preprocessing
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix,accuracy_score,matthews_corrcoef,f1_score
from sklearn.externals import joblib
from matplotlib import pyplot as plt
import seaborn as sns
import joblib
from sklearn import svm
from sklearn.model_selection import GridSearchCV

my_path = '~/Downloads/NEU/bigdata/class'

def mydf_splitter(my_df,num_rows):
    return my_df[:num_rows].copy(),my_df[num_rows:]

def str_to_cat(my_df):
    for p,q in my_df.items(): #my_df.items() is a generator in Python
        if is_string_dtype(q): 
            my_df[p] = q.astype('category').cat.as_ordered()
    return my_df


def mydf_to_nums(my_df, feature, null_status):
    if not is_numeric_dtype(feature):
        my_df[null_status] = feature.cat.codes + 1
        
def mydf_imputer(my_df, feature, null_status, null_table):
    if is_numeric_dtype(feature):
        if pd.isnull(feature).sum() or (null_status in null_table):
            my_df[null_status+'_na'] = pd.isnull(feature)
            filler = null_table[null_status] if null_status in null_table else feature.median()
            my_df[null_status] = feature.fillna(filler)
            null_table[null_status] = filler
    return null_table   

def mydf_preprocessor(my_df, null_table):
    '''null_table  = your table or None'''
    
    if null_table is None: 
        null_table = dict()
    for p,q in my_df.items(): 
        null_table = mydf_imputer(my_df, q, p, null_table)
    for p,q in my_df.items(): 
        mydf_to_nums(my_df, q, p)
    my_df = pd.get_dummies(my_df, dummy_na = True)
    res = [my_df, null_table]
    return res

# preporcess data
my_df = pd.read_csv(f'{my_path}/Titanic_full.csv')
mydf_train_valid, mydf_test = mydf_splitter(my_df,1100)

mydf_train_valid_2 = mydf_train_valid.drop("Cabin",axis = 1)
mydf_train_valid_3 = str_to_cat(mydf_train_valid_2)
mydf_train_valid_4,my_table = mydf_preprocessor(mydf_train_valid_3,null_table = None)

Y = mydf_train_valid_4["Survived"]
X = mydf_train_valid_4.drop(["Survived"],axis = 1)

X_cat = X[['PassengerId', 'Pclass', 'Name', 'Sex', 'SibSp', 'Parch',
       'Ticket', 'Embarked', 'Age_na', 'Fare_na']]
X_con = X.drop(X_cat, axis = 1)

scaler = preprocessing.StandardScaler().fit(X_con)
X_con_sc = pd.DataFrame(scaler.transform(X_con))
X_con_sc.columns = ["Age","Fare"]

df_list = [X_cat,X_con_sc]
X_full = pd.concat(df_list,axis = 1)

X_train,X_valid = mydf_splitter(X_full,900)
Y_train,Y_valid = mydf_splitter(Y,900)

print(X_train.shape,X_valid.shape,Y_train.shape,Y_valid.shape)


# In[2]:

# Use Grid search to tune hyperparameter
parameters = {'weights':['uniform', 'distance'], 'n_neighbors':range(1,20)}
grid_search = GridSearchCV(KNeighborsClassifier(), parameters)
grid_search.fit(X_full, Y)

print("Best score: %0.3f" % grid_search.best_score_)
print("Best parameters set:")
best_parameters = grid_search.best_estimator_.get_params()
for param_name in sorted(parameters.keys()):
    print("\t%s: %r" % (param_name, best_parameters[param_name])) 


# In[12]:

# process the test data
mydf_test1 = mydf_test.drop("Cabin",axis = 1)
mydf_test2 = str_to_cat(mydf_test1)
mydf_test3,my_table1 = mydf_preprocessor(mydf_test2,
                                         null_table = my_table)
Y_t = mydf_test3["Survived"]
X_t = mydf_test3.drop(["Survived"],axis = 1)
X_cat_t = X_t[['PassengerId', 'Pclass', 'Name', 'Sex', 'SibSp', 'Parch',
       'Ticket', 'Embarked', 'Age_na', 'Fare_na']]
X_con_t = X_t.drop(X_cat_t,axis = 1)

X_con_sct = pd.DataFrame(scaler.transform(X_con_t))

X_con_sct.columns = ["Age","Fare"]
X_cat_t.reset_index(inplace = True,drop = False)
X_cat_t.drop("index",inplace = True,axis = 1)

df_list_I = [X_cat_t,X_con_sct]
X_test_I = pd.concat(df_list_I,axis = 1)

# Test the model
knn_model_fin = KNeighborsClassifier(n_neighbors = best_parameters['n_neighbors'], 
                    weights = best_parameters['weights'])
knn_model_fin.fit(X_full,Y)
Y_test_pred = knn_model_fin.predict(X_test_I)
print("Accuracy is %0.3f" % accuracy_score(Y_t,Y_test_pred))
    

