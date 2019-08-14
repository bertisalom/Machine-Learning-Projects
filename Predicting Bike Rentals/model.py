'''
In this project, I'll try to predict the total number of bikes people rented 
in a given hour. I'll predict the 'cnt' column. To accomplish this, I'll create
 a few different machine learning models and evaluate their performance.

Models to be used;

1) Linear Regression
2) Decision Tree
3) Random Forest

I will use RMSE for the model evaluation as this is a regression problem.
'''

# Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error


####################################
# DATA EXPLORATION
####################################

# Read the data
df = pd.read_csv('bike_rental_hour.csv')

# Exploration of data
print(df.head(10))
print(df.info())

# Histogram of target column
plt.hist(df['cnt'], bins=30)
plt.title('Distribution of the total number of bike rentals')
plt.show()


####################################
# FEATURE ENGINEERING
####################################

'''
To enhance the model accuracy, I will reorganize the hr column by grouping the hours like morning, afternoon, 
evening, and night assigning 1, 2, 3 and 4, respectively. 
'''

# Create function to label hr column
def assign_label(hr):
    if (hr >= 0) & (hr < 6):
        return 4
    elif (hr >= 12) & (hr < 18):
        return 1
    elif (hr >= 18) & (hr < 24):
        return 2
    elif (hr >= 6) & (hr < 12):
        return 3

# Apply the function      
df['time_label'] = df['hr'].apply(assign_label)

# Correlation of target column with features
corr = df.corr()
plt.figure(figsize=(15,10))
sns.heatmap(corr, annot=True, linewidths=.5)
plt.title('Correlation heatmap among columns')
plt.show()
target_corr = np.abs(corr['cnt']).sort_values(ascending=False)
print('Correlation with target column:\n', target_corr)

####################################
# TRAIN AND TEST SPLIT
####################################

train = df.sample(frac=0.8, random_state=1)
test = df.loc[~df.index.isin(train.index)]

####################################
# MODEL: LINEAR REGRESSION
####################################    

# Features and target selection
def feature_selection(target_corr, corr_coef=0.3):
    target_corr_features = target_corr.drop(['cnt', 'registered', 'casual']) # drop casual and registered because they constitute cnt column
    features = target_corr_features[target_corr_features > corr_coef].index # Take features with over x correlation coefficient with tacorr
    return features

# Try different correlation coefficient
for corr_coef in np.arange(0,0.5,0.1):   
    features = feature_selection(target_corr, corr_coef)
    target = 'cnt'    
        
# Preparing Model
    lr = LinearRegression()
    
# Fitting the model
    lr.fit(train[features], train[target])
    
# Predicting test
    predictions = lr.predict(test[features])
    
# Evaluating test predictions
    mse = mean_squared_error(test[target], predictions)
    rmse = np.sqrt(mse)
    print('Model: Linear regression\nRMSE value using features over {:.1f} correlation coefficient: {:.2f}\n'.format(corr_coef, rmse))    
    
####################################
# MODEL: DECISION TREE
####################################    
    
# Features and target selection
features = feature_selection(target_corr, corr_coef=0)
target = 'cnt'    
    
# Preparing Model
dtr_rmses = []
dtr_parameters = []
for md in np.arange(3,25,1): # loop over max_depth between 3 and 24
    for msl in np.arange(2,25,1): # loop over min_samples_leaf between 2 and 24
        dtr = DecisionTreeRegressor(max_depth=md, min_samples_leaf=msl, random_state=1)
        
# Fitting the model
        dtr.fit(train[features], train[target])
        
# Predicting test
        predictions = dtr.predict(test[features])
        
# Evaluating test predictions
        mse = mean_squared_error(test[target], predictions)
        rmse = np.sqrt(mse)
        dtr_rmses.append(rmse)
        parameter = (md, msl)
        dtr_parameters.append(parameter)
#       print('Model: Decision Tree\nmax_depth: {} \nmin_samples_leaf: {} \n RMSE value: {}\n'.format(md, msl, rmse))    
best_md, best_msl = dtr_parameters[dtr_rmses.index(min(dtr_rmses))]
print('Model: Decision Tree \nThe optimum model parameters are; \nmax_depth: {} \nmin_samples_leaf: {} \nRMSE: {}\n'.format(best_md, best_msl, min(dtr_rmses)))

####################################
# MODEL: RANDOM FOREST
####################################    

# Features and target selection
features = feature_selection(target_corr, corr_coef=0)
target = 'cnt'    
    
# Preparing Model
rfr_rmses = []
rfr_parameters = []

for md in np.arange(3,25,1): # loop over max_depth between 3 and 24
    for msl in np.arange(2,8,1): # loop over min_samples_leaf between 2 and 7
        for ne in np.arange(10,18,1): # loop over n_estimators between 10 and 17
            rfr = RandomForestRegressor(n_estimators=ne, max_depth=md, min_samples_leaf=msl, random_state=1)
        
# Fitting the model
            rfr.fit(train[features], train[target])
        
# Predicting test
            predictions = rfr.predict(test[features])
        
# Evaluating test predictions
            mse = mean_squared_error(test[target], predictions)
            rmse = np.sqrt(mse)
            rfr_rmses.append(rmse)
            parameter = (md, msl, ne)
            rfr_parameters.append(parameter)
best_md, best_msl, best_ne = rfr_parameters[rfr_rmses.index(min(rfr_rmses))]
print('Model: Random Forest \nThe optimum model parameters are; \nmax_depth: {} \nmin_samples_leaf: {} \nn_estimators: {} \nRMSE: {}\n'.format(best_md, best_msl, best_ne, min(rfr_rmses)))

'''
Based on RMSE values produced by models with various parameters random forest
is the best predictor of the target column.
'''