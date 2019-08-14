# Import libraries
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

'''
In this project, I'll be working with data from the S&P500 Index. 
I will predict the future prices using Linear Regression machine learning 
algorithm. Then, I'll evaluate how good the prediction model is.

'''

# Read the dataset
df = pd.read_csv('sphist.csv')

#######################################
# DATA CLEANING AND FEATURE ENGINEERING
#######################################

# Convert date to datetime and sort data ascending by date
df['Date'] = pd.to_datetime(df['Date'], format='%Y-%m-%d')
df = df.sort_values(by='Date', ascending=True)

'''
Datasets taken from the stock market need to be handled differently than
datasets from other sectors when it comes time to make predictions. In a
normal machine learning model, each row is treated as independent. Stock 
market data is sequential, and each observation comes a day after the previous
observation. Thus, the observations are not all independent, and it can't be
treated as such.

To improve accuracy, I will add these three features as new columns

1) The average price for the past 30 days.
2) The standard deviation of the price over the past 5 days.
3) The ratio between the standard deviation for the past 5 days and the past 365 days.

However, after calculated, I will shift the calculated rows 1 day forward to 
work only with past data when predicting.
'''

# The average price for the past 30 days.
df['avg_30'] = df['Close'].rolling(30).mean()
df['avg_30'] = df['avg_30'].shift(1)

# The standard deviation of the price over the past 5 days.
df['std_5'] = df['Close'].rolling(5).std()
df['std_5'] = df['std_5'].shift(1)

# The ratio between the standard deviation for the past 5 days and the past 365 days.
len_365_days = (df["Date"] <= datetime(year=1951, month=1, day=2)).sum() # calculated how many days in a year in the dataset because there is no weekends in the dataset
df['std_5_over_365'] = df['Close'].rolling(5).std() / df['Close'].rolling(len_365_days).std()
df['std_5_over_365'] = df['std_5_over_365'].shift(1)

# Remove NaN values because of rolling 365 days method
df_up = df.dropna(axis=0).copy()


#######################################
# TRAIN AND TEST SPLIT
#######################################

train = df_up[df_up['Date'] < datetime(year=2013, month=1, day=1)].copy()
test = df_up[df_up['Date'] >= datetime(year=2013, month=1, day=1)].copy()

#######################################
# MODEL: LINEAR REGRESSION
#######################################

# Features and target selection
features = ['avg_30', 'std_5', 'std_5_over_365']
target = 'Close'

# Preparing Model
lr = LinearRegression()

# Fitting the model
lr.fit(train[features], train[target])

# Predicting test
predictions = lr.predict(test[features])

# Evaluating test predictions
mse = mean_squared_error(test[target], predictions)
rmse = np.sqrt(mse)
print('3 features prediction error: ', rmse)


#######################################
# MODEL IMPROVEMENT
#######################################

# Adding more features

# The average price for the past 30 days.
df['avg_30'] = df['Close'].rolling(30).mean()
df['avg_30'] = df['avg_30'].shift(1)

# The standard deviation of the price over the past 5 days.
df['std_5'] = df['Close'].rolling(5).std()
df['std_5'] = df['std_5'].shift(1)

# The ratio between the standard deviation for the past 5 days and the past 365 days.
len_365_days = (df["Date"] <= datetime(year=1951, month=1, day=2)).sum() # calculated how many days in a year in the dataset because there is no weekends in the dataset
df['std_5_over_365'] = df['Close'].rolling(5).std() / df['Close'].rolling(len_365_days).std()
df['std_5_over_365'] = df['std_5_over_365'].shift(1)

# The ratio between the average volume for the past five days, and the average volume for the past year.
df['avg_volume_5_days_over_year'] = df['Volume'].rolling(5).mean() / df['Volume'].rolling(len_365_days).mean() 
df['avg_volume_5_days_over_year'] = df['avg_volume_5_days_over_year'].shift(1)

# The ratio between the highest price in the past year and the current price
df['max_price_year_over_current_price'] = df['Adj Close'].rolling(len_365_days).max() / df['Adj Close'].rolling(1).mean()
df['max_price_year_over_current_price'] = df['max_price_year_over_current_price'].shift(1)

# Month
df['month'] = df['Date'].dt.month
df['month'] = df['month'].shift(1)

# Year
df['year'] = df['Date'].dt.year
df['year'] = df['year'].shift(1)


# Remove NaN values because of rolling 365 days method
df_up = df.dropna(axis=0).copy()


#######################################
# TRAIN AND TEST SPLIT
#######################################

train = df_up[df_up['Date'] < datetime(year=2013, month=1, day=1)].copy()
test = df_up[df_up['Date'] >= datetime(year=2013, month=1, day=1)].copy()

#######################################
# MODEL: LINEAR REGRESSION
#######################################

# Features and Target selection
features = ['avg_30', 'std_5', 'std_5_over_365', 'avg_volume_5_days_over_year', 'max_price_year_over_current_price', 'month', 'year']
target = 'Close'

# Preparing Model
lr = LinearRegression()

# Fitting the model
lr.fit(train[features], train[target])

# Predicting test
predictions = lr.predict(test[features])

# Evaluating test predictions
mse = mean_squared_error(test[target], predictions)
rmse = np.sqrt(mse)
print('6 features prediction error: ', rmse)

'''
Further Improvement

Accuracy would improve greatly by making predictions only one day ahead. 
For example, train a model using data from 1951-01-03 to 2013-01-02, make 
predictions for 2013-01-03, and then train another model using data from 
1951-01-03 to 2013-01-03, make predictions for 2013-01-04, and so on. This more
 closely simulates what you'd do if you were trading using the algorithm.

Try other techniques: like a random forest, and observe if they perform better.

You can also incorporate outside data, such as the weather in New York City 
(where most trading happens) the day before, and the amount of Twitter activity
 around certain stocks.

You can also make the system real-time by writing an automated script to 
download the latest data when the market closes, and make predictions for the 
next day.

Finally, you can make the system "higher-resolution". You're currently making 
daily predictions, but you could make hourly, minute-by-minute, or second by 
second predictions. This will require obtaining more data, though. You could 
also make predictions for individual stocks instead of the S&P500.
'''

