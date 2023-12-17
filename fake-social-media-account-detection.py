from sys import displayhook
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
import pdpipe as pdp
import pickle
import os

for dirname, _, filenames in os.walk('H:\code\detection'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

import warnings
warnings.filterwarnings('ignore')



import warnings
warnings.filterwarnings('ignore')
pd.options.display.max_colwidth=300
np.set_printoptions(suppress=True) 
pd.options.display.float_format = '{:.2f}'.format

pd.set_option('display.max_columns', None)

df_train = pd.read_csv("H:/code/detection/social-media-train/social_media_train.csv",index_col=[0])
displayhook(df_train.head())
displayhook(df_train.info())


# Each line of df_train represents a user or user account.
# 

# 
# 

# In[16]:


# Dataset Description
data_dict = pd.read_csv('H:/code/detection/fake-account-data-dict/fake_account__data_dict.csv', index_col = 'No.')
data_dict


# In[17]:


num_cols = ['ratio_numlen_username','len_fullname','ratio_numlen_fullname',
            'len_desc','num_posts','num_followers',
              'num_following']

cat_cols = [col for col in df_train.columns.values.tolist() if col not in num_cols]
cat_cols


# In[18]:


# Check descriptive statistics
df_train[num_cols].describe()


# In[19]:


# percent of fake n not fake
plt.figure(figsize=(15,6))
fake_share = df_train["fake"].value_counts()
mylabel=["Not fake(0)","fake(1)"]
colors = ['#99ff99','#ff9999']
plt.pie(fake_share,
        labels=mylabel,autopct="%1.1f%%",colors=colors,
        textprops={'fontsize': 16})
plt.axis("equal");


# Target category is strongly balanced.
# 

# In[20]:


# Check the percentage of the missing values

percent_missing = df_train.isnull().sum() * 100 / len(df_train)
missing_value_df = pd.DataFrame({'percent_missing (%)': percent_missing})
missing_value_df.sort_values('percent_missing (%)', ascending=False)


# In[21]:


# Pair plot oapif numerical values with fake account information
pp_cols = []
pp_cols = num_cols + ['fake']
ax=sns.pairplot(df_train[pp_cols], hue="fake",corner=True);
plt.style.use('fivethirtyeight')
ax.fig.suptitle("Pair Plot of Characteristics")


# In[22]:


# Categorical data
displayhook(df_train.loc[:, cat_cols].head())
print('----------------------')

# Unique values
for col in cat_cols:
    unique_values = df_train.loc[:, col].unique()
    print("\nColumn name: {}\nUnique values: {}".format(col, unique_values))  


# In[23]:


# Define function label_encoding
def label_encoding(df):
    '''
    Function label_encoding() trnasforms categorical features
    represented by strings to binary features containing only
    0s and 1s which can be used for machine learning models.
    Input:
        DataFrame of features
    Output:
        New DataFrame with binary features    
    '''
    # label encoding
    dict_label_encoding = {'Yes': 1, 'No': 0}
    df.loc[:, 'profile_pic'] = df.loc[:, 'profile_pic'].replace(dict_label_encoding)
    df.loc[:, 'extern_url'] = df.loc[:, 'extern_url'].replace(dict_label_encoding)
    df.loc[:, 'private'] = df.loc[:, 'private'].replace(dict_label_encoding)

    # one-hot encoding
    onehot = pdp.OneHotEncode(["sim_name_username"], drop_first=False)
    # fit and transform on train set
    df = onehot.fit_transform(df)
    return df       

# Applay function label_encoding
df_train = label_encoding(df_train)
df_train.head()


# In[ ]:


## without regularisation


# In[24]:


from sklearn.linear_model import LogisticRegression


# In[25]:


features_train = df_train.iloc[:, 1:]
target_train = df_train.loc[:, 'fake']

model_log = LogisticRegression(solver='lbfgs', max_iter=10000, C=1e42, random_state=42)


# In[26]:


model_log.fit(features_train, target_train)


# In[75]:


## with regularisation


# In[27]:


from sklearn.preprocessing import StandardScaler 


# In[28]:


scaler = StandardScaler()


# In[29]:


features_train_scaled = scaler.fit_transform(features_train)


# In[30]:


model_reg = LogisticRegression(solver='lbfgs', max_iter=10000, C=0.5, random_state=42)


# In[31]:


model_reg.fit(features_train_scaled, target_train)


# # 5) Model Performance Evaluation 
# 
# We use test data from social_media_test.csv for initial classification performance:

# In[33]:


# Read data
df_test = pd.read_csv("H:/code/detection/social-media-test/social_media_test.csv",index_col=[0])
df_test = label_encoding(df_test)
df_test.head()


# In[34]:


# Feature matrix and target vector
features_test = df_test.drop('fake',axis=1)
target_test = df_test['fake']

# Without regularisation
# predict target values from model
target_test_pred_log = model_log.predict(features_test)

# model evaluation
from sklearn.metrics import precision_score, recall_score
precision_log = precision_score(target_test, target_test_pred_log)
recall_log = recall_score(target_test, target_test_pred_log)

# print
# print('Precision of model without regularisation: ', precision_log)
# print('Recall of model without regularisation: ', recall_log)


# In[35]:


# With Regularisation
from sklearn.preprocessing import StandardScaler


# In[36]:


scaler = StandardScaler()


# In[37]:


print(scaler.fit(features_test))


# In[38]:


features_test_scaled = scaler.transform(features_test)


# In[39]:


target_test_pred_reg = model_reg.predict(features_test_scaled)


# In[40]:


precision_reg = precision_score(target_test, target_test_pred_reg)
recall_reg = recall_score(target_test, target_test_pred_reg)


# In[41]:


# print('Precision of model with regularisation: ', precision_reg)
# print('Recall of model with regularisation: ', recall_reg)

# In[42]:


# module import
from sklearn.metrics import roc_curve

# calculate roc curve values
def roc_curve_values(model,features,target):
    '''
    Function roc_curve_values estimates the probability
    and return roc_curve values as output.
    
    Input:
        model, feautes as dataframe, target values
    Output:
        False positive rate, recall, target_test_pred_proba
    '''
    # calculate probability
    target_test_pred_proba = model.predict_proba(features) 
    
    # calculate roc curve values
    false_positive_rate, recall, threshold = roc_curve(target,
                                                       target_test_pred_proba[:,1],
                                                       drop_intermediate=False)
    
    return false_positive_rate, recall, target_test_pred_proba    


# In[43]:
# Apply function roc_curve_values for model without regularization
false_positive_rate_log, recall_log, target_test_pred_proba_log  = roc_curve_values(model_log,features_test,target_test)


# In[44]:
# Apply function roc_curve_values for model with regularization
false_positive_rate_reg, recall_reg, target_test_pred_proba_reg  = roc_curve_values(model_reg,features_test,target_test)


# In[45]:


def roc_curve_plot(false_positive_rate,recall,label):
    '''
    Function roc_curve_plot plots ROC
    Input:
        false_positive_rate, recall, label: model type
    Output:
        ROC plot    
    '''
    plt.style.use('fivethirtyeight')
    fig,ax=plt.subplots()
    
    # Reference lines
    # Blue diagonal
    ax.plot([0, 1], ls = "--", label='random model')  
    # Grey vertical
    ax.plot([0, 0], [1, 0], c=".7", ls='--', label='ideal model') 
    # Grey horizontal
    ax.plot([1, 1], c=".7", ls='--')  
    
    # ROC curve
    ax.plot(false_positive_rate,recall, label = label)
    
    # labels
    ax.set_title("Receiver Operating Characteristic")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("Recall")
    ax.legend()


# In[46]:

roc_curve_plot(false_positive_rate_log,recall_log,'model_log')

# In[47]:

roc_curve_plot(false_positive_rate_reg,recall_reg,'model_reg')

# In[48]:


# Import roc_auc_score
from sklearn.metrics import roc_auc_score
# print('roc_auc_score for model without regularization', roc_auc_score(target_test, target_test_pred_proba_log[:, 1]))
roc_auc_score(target_test, target_test_pred_proba_log[:, 1])
roc_auc_score(target_test, target_test_pred_proba_reg[:, 1])
print('##########################################################')
# print('roc_auc_score for model with regularization', roc_auc_score(target_test, target_test_pred_proba_reg[:, 1]))

# In[49]:

from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV

# Define pipeline
pipeline_log = Pipeline([('scaler',StandardScaler()),('classifier',LogisticRegression(solver='saga',
                                                                  max_iter=10000, 
                                                                  random_state=42))])

C_values = np.geomspace(start=0.001, stop=1000, num=14)
search_space_grid = [{'classifier__penalty': ['l1', 'l2'],
                      'classifier__C': C_values}]

model_grid = GridSearchCV(estimator=pipeline_log,
                          param_grid=search_space_grid,
                          scoring='roc_auc',
                          cv=5,
                          n_jobs=-1)

# Model fitting
model_grid.fit(features_train, target_train)

# Print best estimater and score
# print(model_grid.best_estimator_)
# print(model_grid.best_params_)
# print(model_grid.best_score_)

# In[50]:
# Apply function roc_curve_values for model with regularization
false_positive_rate_grid, recall_grid, target_test_pred_proba  = roc_curve_values(model_grid,features_test,target_test)
roc_curve_plot(false_positive_rate_grid,recall_grid,'model_grid')

# In[51]:

# Calculated roc_auc_score
target_test_pred_proba = model_grid.predict_proba(features_test)
roc_auc_score(target_test, target_test_pred_proba[:, 1])


# In[52]:
# Read data
df_aim = pd.read_csv("H:/code/detection/social-media-aim/social_media_aim.csv",index_col=[0])
df_aim = label_encoding(df_aim)
features_aim = df_aim.copy()
# print(features_aim)

# Apply Prediction 
df_aim.loc[:, 'fake_pred_proba'] = model_grid.predict_proba(features_aim)[:, 1]
df_aim.loc[:, 'fake_pred'] = model_grid.predict(features_aim)
type(df_aim)

displayhook(df_aim)
# plt.show()

with open('fake-social-media-account-detection.pkl', 'wb') as file:
    pickle.dump(model_grid, file)

