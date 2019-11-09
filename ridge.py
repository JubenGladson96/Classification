import numpy as np
import pandas as pd

data = pd.read_csv('C:\\Users\\Nithin\\Downloads\\SSN\Machine\\house\\train.csv',index_col='Id')
X=data.drop(['SalePrice'],axis=1)
y=data.SalePrice
X_test = transform_X(pd.read_csv('C:\\Users\\Nithin\\Downloads\\SSN\\Machine\\house\\test.csv', index_col='Id'))
# Count the missing value and their share in the data
total=X.isnull().sum().sort_values(ascending=False)
percent = (X.isnull().sum()/X.isnull().count()).sort_values(ascending = False)
missing_data = pd.concat([total, percent], axis =1, keys=['Total', 'Percentage'])
# Drop the variables with more than 10 percent of missing values in both
X = X.drop(missing_data[missing_data.Percentage > 0.1].index, axis = 1)
X_test = X_test.drop(missing_data[missing_data.Percentage > 0.1].index, axis = 1)
#Drop the variables that were collected after the dependent variable to prevent for data leakage. Also do this for the test data
X = X.drop(['SaleType','SaleCondition','Id','MoSold', 'YrSold'], axis = 1)
X_test = X_test.drop(['SaleType','SaleCondition','Id','MoSold', 'YrSold'], axis = 1)
# As we are only going to use the numerical data for building our model, we create this model
numerical_cols = [cname for cname in X.columns if X[cname].dtype in ['int64', 'float64']]
# Subset the numerical data from the train and test data
X = X[numerical_cols]
X_test = X_test[numerical_cols]
# To deal with the remaining missing values in these numerical collumns we use the SimpleImputer function and impute the missing 
# values with median of their respective column
from sklearn.preprocessing import Imputer
numerical_transformer = Imputer(strategy='median')
X = numerical_transformer.fit_transform(X)
X_test = numerical_transformer.fit_transform(X_test)
# Importing the functions to select the K best features and using it to find the 15 best variables
from sklearn.feature_selection import SelectKBest, chi2
Classifier = SelectKBest(chi2, k=20)
Classifier.fit(X, y)
mask = Classifier.get_support(indices = True)
X = X[:,mask]
X_test = X_test[:,mask]
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import Ridge
model = Ridge()
parameters = {'alpha':[1e-15, 1e-10, 1e-8, 1e-4, 1e-3, 1e-2, 1, 5, 10 ,15, 20]}
ridge_tuner = GridSearchCV(model, parameters, scoring = 'neg_mean_squared_error', cv=5, verbose=5)
ridge_tuner.fit(X,y)
# Printing out the best parameters
PARAMS = ridge_tuner.best_params_
print(str(PARAMS))
print(ridge_tuner.best_score_)
# Creating of the model and setting the hyperparameters
model = Ridge(**PARAMS)
print('Model created')
# Fitting of the model
model.fit(X,y)
print('Model fitted')
# Predicting of the data using the fitted model
predictions = model.predict(X_test)
print('Predictions made')

# Creating the output file
output = pd.DataFrame({'Id': pd.read_csv('test.csv').Id,
                       'SalePrice': predictions})
output.to_csv('submission.csv', index=False)