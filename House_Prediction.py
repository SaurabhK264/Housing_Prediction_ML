#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')


# In[2]:


data=pd.read_csv('House_price.csv')
data.head()


# In[3]:


data.shape


# In[4]:


data.nunique()


# In[5]:


data.info()


# * In this dataset there are 38 numerical variables(int+float) and 43 categorical variables.
# * MSSubClass, OverallQual, OverallCond should be an object datatype

# In[6]:


data.isna().sum()


# In[7]:


data.isna().sum().sum()


# In[8]:


null_counts=data.isna().sum()


# In[9]:


variables_with_null=null_counts[null_counts>0]


# In[10]:


variables_with_null


# In[11]:


#Filling numerical null values with the mean 
numerical_variables=['LotFrontage','MasVnrArea','GarageYrBlt']
for i in numerical_variables:
    data[i].fillna(data[i].mean(),inplace=True)


# In[12]:


#filling categorical null values with 'None'
categorical_variables=['Alley','MasVnrType','BsmtQual','BsmtCond','BsmtExposure','BsmtFinType1','BsmtFinType2','Electrical',
                      'FireplaceQu','GarageType','GarageFinish','GarageQual','GarageCond','PoolQC','Fence','MiscFeature']
for i in categorical_variables:
    data[i].fillna('None',inplace=True)


# In[13]:


data.isna().sum().sum()


# In[14]:


data.info()


# In[15]:


object_columns=data.select_dtypes(include='object').columns.tolist()
numericals_columns=data.select_dtypes(include=['int','float']).columns.tolist()


# In[16]:


print("object columns:", object_columns)
print('\n')
print("numerical columns:",numericals_columns)


# In[17]:


data.describe().T


# In[18]:


data.nunique()


# In[19]:


for i in object_columns:
    print(i)
    print(data[i].unique())
    print('\n')


# In[20]:


for i in object_columns:
    print(data[i])
    print(data[i].value_counts())
    print('\n')


# # Univariate Anaysis
#    * Categorical Variables
#    * Numerical Variable

# In[21]:


#let's analyse categorical variable
for i in object_columns:
    print('Distribution of',i)
    plt.figure(figsize=(6,3))
    sns.countplot(data[i],data=data)
    plt.xticks(rotation=-45)
    plt.show()


# In[22]:


#let's analyse numerical variables
for i in numericals_columns:
    print('Box Plot of:',i)
    plt.figure(figsize=(10,4))
    sns.boxplot(data[i],orient='vertical')
    plt.show()


# * For many numerical variables, there are many outliers present so it should be removed.

# In[23]:


#let's see whether the values are normally distributed or not
for i in numericals_columns:
    print('KDE plot of:',i)
    plt.figure(figsize=(10,4))
    sns.distplot(data[i],kde=True)
    plt.show()


# # Bivariate Analysis
#  * Numerical-Numerical.
#  * Numerical-Categorical.

# In[228]:


#numerical-numerical
for i in numericals_columns:
    for j in numericals_columns:
        if i !=j:
            plt.figure(figsize=(10,4))
            sns.lineplot(x=data[j],y=data[i],data=data,ci=None)
            plt.show()


# * As we can see there are many lineplots, so it would be a difficult task to visualize one by one.

# In[229]:


for i in numericals_columns:
    for j in numericals_columns:
        if i !=j:
            print('Scatterplot of:',i)
            plt.figure(figsize=(10,4))
            sns.scatterplot(x=data[j],y=data[i],data=data)
            plt.show()


# In[230]:


#categorical-categorical
for i in numericals_columns:
    for j in object_columns:
        plt.figure(figsize=(10,4))
        sns.barplot(x=data[j],y=data[i],data=data)
        plt.show()


# In[231]:


for i in numericals_columns:
    for j in object_columns:
        plt.figure(figsize=(10,4))
        sns.boxplot(x=data[j],y=data[i],data=data)
        plt.show()


# In[232]:


data.corr()


# * Now we are going to plot heatmap to visualize correlation in a better way.

# In[233]:


plt.figure(figsize=(30,20))
sns.heatmap(data.corr(),cmap='coolwarm',fmt='.2f',annot=True)
plt.title('Correlation Plot')
plt.show()


# * As we can visualize from this heatmap that 35snPorch, ScreenPorch, PoolArea, MiscVal, MoSold, YrSold varibales has low correlation with other varisbles.

# In[234]:


df=data.copy()


# In[235]:


df.head()


# In[236]:


plt.figure(figsize=(10,4))
sns.histplot(df['SalePrice'],kde=True)
plt.show()


# * As we can see, the distribution of the target variable is right skewed. So we need to do log transformation and see how it works.

# In[237]:


df['SalePrice']=np.log(df['SalePrice'])


# In[238]:


plt.figure(figsize=(10,4))
sns.histplot(df['SalePrice'],kde=True)
plt.show()


# In[239]:


from scipy import stats


# In[240]:


fig=plt.figure()
res=stats.probplot(df['SalePrice'],plot=plt)


# In[241]:


#skewness
skewness=df[numericals_columns].skew()
print(skewness)


# * As we can see there are many numerical variables who have very high skew value.

# In[242]:


#let's transform right skewed values
skewed_columns=skewness[(skewness>1)]
skewed_columns


# In[243]:


skewed_features=['MSSubClass','LotFrontage','LotArea','MasVnrArea','BsmtFinSF1','BsmtFinSF1','TotalBsmtSF','1stFlrSF',
                'LowQualFinSF','GrLivArea','BsmtHalfBath','KitchenAbvGr','WoodDeckSF','OpenPorchSF','EnclosedPorch','3SsnPorch',
                'ScreenPorch','PoolArea','MiscVal']


# In[244]:


for i in skewed_features:
    df[i]=np.log1p(df[i])


# In[245]:


transformed_features=['MSSubClass','LotFrontage','LotArea','MasVnrArea','BsmtFinSF1','BsmtFinSF1','TotalBsmtSF','1stFlrSF',
                'LowQualFinSF','GrLivArea','BsmtHalfBath','KitchenAbvGr','WoodDeckSF','OpenPorchSF','EnclosedPorch','3SsnPorch',
                'ScreenPorch','PoolArea','MiscVal']


# In[246]:


for i in transformed_features:
    plt.figure(figsize=(10,6))
    sns.histplot(df[i],kde=True)
    plt.show()


# In[247]:


#skewness
skewness=df[numericals_columns].skew()
print(skewness)


# # Model Building

# In[248]:


df=df.drop('Id',axis=1)


# In[249]:


#encoding the object columns
df=pd.get_dummies(df,columns=object_columns,drop_first=True)


# **Segragting Independent and dependent variables**

# In[250]:


x=df.drop('SalePrice',axis=1)
y=df['SalePrice']


# **Splitting the data into train and test**

# In[251]:


from sklearn.model_selection import train_test_split as tts
train_x,test_x,train_y,test_y=tts(x,y,test_size=0.2,random_state=42)


# **Linear Regression**

# In[356]:


from sklearn.linear_model import LinearRegression as LR
from sklearn.metrics import mean_squared_error,r2_score


# In[357]:


linear_model=LR()


# In[358]:


#training the model
linear_model.fit(train_x,train_y)


# In[359]:


#predicting the model
y_pred=linear_model.predict(test_x)
y_pred


# In[360]:


mse=mean_squared_error(test_y,y_pred)
rmse=np.sqrt(mse)
r2_score_LR=r2_score(test_y,y_pred)
print('Mean Squared Error',mse)
print('Room Mean Squared Error',rmse)
print('Accuracy Score',r2_score_LR)


# **Regularization**
#  * Ridge and Lasso

# In[257]:


from sklearn.linear_model import Ridge,Lasso


# In[258]:


Lasso_Model=Lasso()
Ridge_Model=Ridge()


# In[259]:


Lasso_Model.fit(train_x,train_y)


# In[260]:


lasso_predict_y=Lasso_Model.predict(test_x)


# In[261]:


Ridge_Model.fit(train_x,train_y)


# In[263]:


ridge_predict_y=Ridge_Model.predict(test_x)


# In[267]:


from sklearn.metrics import r2_score
mse_lasso=mean_squared_error(test_y,lasso_predict_y)
rmse_lasso=np.sqrt(mse_lasso)
r2_score_lasso=r2_score(test_y,lasso_predict_y)


mse_ridge=mean_squared_error(test_y,ridge_predict_y)
rmse_ridge=np.sqrt(mse_ridge)
r2_score_ridge=r2_score(test_y,ridge_predict_y)

print('Lasso Regression- Mean Squared Error:',mse_lasso)
print('Lasso Regression- Root Mean Squared Error:',rmse_lasso)
print('Lasso Regression- Accuracy Score:',r2_score_lasso)
print('\n')
print('Ridge Regression- Mean Squared Error:',mse_ridge)
print('Ridge Regression- Root Mean Squared Error:',rmse_ridge)
print('Ridge Regression- Accuracy Score:',r2_score_ridge)


# **Decision Tree**

# In[268]:


from sklearn.tree import DecisionTreeRegressor


# In[278]:


Decision_Tree_Model=DecisionTreeRegressor(random_state=42,max_depth=4)


# In[279]:


Decision_Tree_Model.fit(train_x,train_y)


# In[280]:


decision_tree_predict_y=Decision_Tree_Model.predict(test_x)


# In[281]:


decision_tree_predict_y


# In[282]:


from sklearn.metrics import mean_squared_error,r2_score


# In[283]:


mse_decision_tree=mean_squared_error(test_y,decision_tree_predict_y)
rmse_decision_tree=np.sqrt(mse_decision_tree)
r2_score_decision_tree=r2_score(test_y,decision_tree_predict_y)

print('Decision Tree- Mean Squared Error:',mse_decision_tree)
print('Decision Tree- Root Mean Squared Error:',rmse_decision_tree)
print('Decision Tree- Accuracy Score:',r2_score_decision_tree)


# # Ensemble Techniques
#  * Bagging
#    * Random Forest
#  * Boosting  
#    * Gradient Boosting Algo
#    * XGBoost
#    * Adaptive Boost

# **Random Forest**

# In[284]:


from sklearn.ensemble import RandomForestRegressor


# In[313]:


random_forest_model=RandomForestRegressor(max_depth=12,random_state=42,n_estimators=10)


# In[314]:


random_forest_model.fit(train_x,train_y)


# In[315]:


random_forest_predict_y=random_forest_model.predict(test_x)


# In[316]:


from sklearn.metrics import mean_squared_error,r2_score


# In[317]:


mse_random_forest=mean_squared_error(test_y,random_forest_predict_y)
rmse_random_forest=np.sqrt(mse_random_forest)
r2_score_random_forest=r2_score(test_y,random_forest_predict_y)

print('Random Forest- Mean Squared Error:',mse_random_forest)
print('Random Forest- Root Mean Squared Error:',rmse_random_forest)
print('Random Forest- Accuracy Score:',r2_score_random_forest)


# **Gradient Boosting**

# In[318]:


from sklearn.ensemble import GradientBoostingRegressor


# In[321]:


gradient_boosting_model=GradientBoostingRegressor(random_state=42)


# In[322]:


gradient_boosting_model.fit(train_x,train_y)


# In[324]:


gradient_boosting_predict_y=gradient_boosting_model.predict(test_x)


# In[325]:


from sklearn.metrics import mean_squared_error,r2_score
mse_gradient_boosting=mean_squared_error(test_y,gradient_boosting_predict_y)
rmse_gradient_boosting=np.sqrt(mse_gradient_boosting)
r2_score_gradient_boosting=r2_score(test_y,gradient_boosting_predict_y)

print('Gradient Boosting- Mean Squared Error:',mse_gradient_boosting)
print('Gradient Boosting- Root Mean Squared Error:',rmse_gradient_boosting)
print('Gradient Boosting- Accuracy Score:',r2_score_gradient_boosting)


# **XGBoosting**

# In[328]:


from xgboost import XGBRegressor


# In[329]:


XGBoost_Model=XGBRegressor(random_state=42)


# In[331]:


XGBoost_Model.fit(train_x,train_y)


# In[333]:


XGBoost_predict_y=XGBoost_Model.predict(test_x)


# In[340]:


from sklearn.metrics import mean_squared_error,r2_score
mse_XGBoost=mean_squared_error(test_y,XGBoost_predict_y)
rmse_XGBoost=np.sqrt(mse_XGBoost)
r2_score_XGBoost=r2_score(test_y,XGBoost_predict_y)

print('XG Boosting- Mean Squared Error:',mse_XGBoost)
print('XG Boosting- Root Mean Squared Error:',rmse_XGBoost)
print('XG Boosting- Accuracy Score:',r2_score_XGBoost)


# **Adaptive Boosting**

# In[336]:


from sklearn.ensemble import AdaBoostRegressor


# In[338]:


Adaptive_Model=AdaBoostRegressor(random_state=42)


# In[339]:


Adaptive_Model.fit(train_x,train_y)


# In[341]:


ada_model_predict_y=Adaptive_Model.predict(test_x)


# In[342]:


from sklearn.metrics import mean_squared_error,r2_score
mse_ada_boost=mean_squared_error(test_y,ada_model_predict_y)
rmse_ada_boost=np.sqrt(mse_ada_boost)
r2_score_ada_boost=r2_score(test_y,ada_model_predict_y)

print('Adaptive Boosting- Mean Squared Error:',mse_ada_boost)
print('Adaptive Boosting- Root Mean Squared Error:',rmse_ada_boost)
print('Adaptive- Accuracy Score:',r2_score_ada_boost)


# In[361]:


r2_score=[r2_score_LR,r2_score_ridge,r2_score_lasso,r2_score_decision_tree,r2_score_random_forest,r2_score_gradient_boosting,
         r2_score_XGBoost,r2_score_ada_boost]
algo=['Linear Regression','Ridge','Lasso','Decision Tree','Random Forest','Gradient Boosting','XG Boosting','Adaptive Boosting']
r2_score


# **Comparing All the Models**

# In[364]:


plt.figure(figsize=(10,6))
sns.barplot(x=algo,y=r2_score)
plt.xlabel('Algorathm')
plt.ylabel('Accuracy Score')
plt.title('Comparing All the Models')
plt.xticks(rotation=45)
plt.show()


# * We can see Ridge model is performing best in all these models. 
# * This could be the result of overfitting.

# # Cross Validation
#    * As we see the Ridge model has the best accuracy scores in all the models. But this can be result of overfitting. In order to find out the real best model, we will cross validate the models and compare their mean accuracy scores.

# In[365]:


from sklearn.model_selection import cross_val_score


# In[367]:


lg_scores=cross_val_score(linear_model,x,y,cv=10) #cross validation model
print(lg_scores) #accuracy score of all cross validation cycle
print(f'mean of accuracy score for Linear Regression model is {lg_scores.mean()*100}\n')

ridge_scores=cross_val_score(Ridge_Model,x,y,cv=10)
print(ridge_scores)
print(f'mean of accuracy score for ridge model is {ridge_scores.mean()*100}\n')

lasso_scores=cross_val_score(Lasso_Model,x,y,cv=10)
print(lasso_scores)
print(f'mean of accuracy for Lasso model is {lasso_scores.mean()*100}\n')

dt_scores=cross_val_score(Decision_Tree_Model,x,y,cv=10)
print(dt_scores)
print(f'mean of accuracy score for Decision Tree Model is {dt_scores.mean()*100}\n')

rfm_scores=cross_val_score(random_forest_model,x,y,cv=10)
print(rfm_scores)
print(f'mean of accuracy score for Random Forest Tree model is {rfm_scores.mean()*100}\n')

gb_scores=cross_val_score(gradient_boosting_model,x,y,cv=10)
print(gb_scores)
print(f'mean of accuracy score for Gradient Boosting model is {gb_scores.mean()*100}\n')

xgb_scores=cross_val_score(XGBoost_Model,x,y,cv=10)
print(xgb_scores)
print(f'mean of accuracy score for XG Boosting model is {xgb_scores.mean()*100}\n')

ada_scores=cross_val_score(Adaptive_Model,x,y,cv=10)
print(ada_scores)
print(f'mean of accuracy score for Adaptive Model is {ada_scores.mean()*100}\n')


# In[372]:


l1=['Linear Regression','Lasso','Ridge','Decision Tree','Random Forest','Gradient Boosting','XG Boosting','Adaptive Boosting']
l2=[r2_score_LR*100,r2_score_lasso*100,r2_score_ridge*100,r2_score_decision_tree*100,r2_score_random_forest*100,
   r2_score_gradient_boosting*100,r2_score_XGBoost*100,r2_score_ada_boost*100]
l3=[lg_scores.mean()*100,ridge_scores.mean()*100,lasso_scores.mean()*100,dt_scores.mean()*100,rfm_scores.mean()*100,
   gb_scores.mean()*100,xgb_scores.mean()*100,ada_scores.mean()*100]
for i in range(0,8):
    temp=(l2[i]-l3[i])
    print(l1[i],temp)


# * After the cross validation, we see that the least difference between mean accuracy and total accuracy is given by Random Forest model, so we will build our final model on Random forest only.

# # Hyperparameter Tuning
#    * We have selected the Random Forest model as the best possible model for this case study, we will now tune the parameters of this model to get the best possible results.

# In[375]:


from sklearn.model_selection import RandomizedSearchCV


# In[405]:



#number of trees in random forest
n_estimators=[10,20,30,50,100,150,200]

#number of features
max_features=['auto','sqrt']

#maximum depth of the tree
max_depth=[4,5,6,7,8,9,10,11,12,13]

#minimum number of samples required to split a node
min_sample_split=[2,5,10]

#minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 4]

#method of selecting samples for training each tree
bootstrap = [True, False]

# Create the random grid
random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_sample_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}


# In[407]:


rf=RandomForestRegressor()

random_search=RandomizedSearchCV(estimator = rf, param_distributions = random_grid, 
                                n_iter = 100, cv = 5, verbose=2)
random_search.fit(train_x,train_y)
print(random_search.best_score_)
print(random_search.best_params_)


# * The best parameters for the model are: {'n_estimators': 200, 'min_samples_split': 2, 'min_samples_leaf': 2, 'max_features': 'sqrt', 'max_depth': 12, 'bootstrap': False}, we will use this parameter to build our model.

# In[408]:


from sklearn.metrics import r2_score
rf=RandomForestRegressor(n_estimators= 200, min_samples_split= 2, min_samples_leaf= 2, 
       max_features='sqrt', max_depth= 12, bootstrap= False)
rf.fit(train_x,train_y)
rf_predict_y=rf.predict(test_x)
print(rf.score(train_x,train_y))
print(r2_score(test_y,rf_predict_y))


# # Model Evaluation
#    * We have built the best models after their cross validation and tuning. It is now time to evaluate the models performance using the evaluation metrics.

# In[409]:


from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import mean_absolute_error as mae


# In[410]:


plt.style.use('ggplot')


# In[411]:


print("Accuracy Score of Random Forest Regressor model is", r2_score(test_y, rf_predict_y)*100)
print("The mean absolute error of the fitted model is", mae(test_y, rf_predict_y))
print("The mean squared error of the fitted model is", mse(test_y, rf_predict_y))
print("The root mean squared error of the fitted model is", np.sqrt(mse(test_y, rf_predict_y)))

plt.figure(figsize = (10,6))
plt.title("RFR Model- Prediction vs Actual Values", fontsize = 14)
plt.scatter(x = test_y, y = rf_predict_y, color = 'r')
plt.plot(test_y, test_y, color = 'b')
plt.show()


# # House Price Prediction

# In[412]:


rf_predict_y=rf.predict(test_x)
rf_predict_y


# In[414]:


final_y=rf_predict_y


# In[415]:


final_y


# In[416]:


# Natural log and back to normal value using built-in numpy exp() function
final_y=np.exp(final_y)
final_y


# # Saving Prediction

# In[417]:


ans_sub = pd.DataFrame(data=final_y, columns=['Predicted SalePrice'])
writer = pd.ExcelWriter('House Sale Price Prediction.xlsx', engine='xlsxwriter')
ans_sub.to_excel(writer,sheet_name='House Sale Prices', index=False)
writer.save()


# In[418]:


ans_sub


# In[ ]:





# In[ ]:




