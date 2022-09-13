import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm

LE = preprocessing.LabelEncoder()
OHE = preprocessing.OneHotEncoder()
regressor = LinearRegression()

base = pd.read_csv()
base2 = base.apply(LE.fit_transform)

data_math = base.iloc[:,5:6]
#data_read = base.iloc[:,[0,1,2,3,4,6]].values
#data_write = base.iloc[:,[0,1,2,3,4,7]].values

gender = base2.iloc[:,0:1] #1 = female, 0 = male
lunch = base2.iloc[:,3:4] #1 = standard, 0 = free/reduced
course = base2.iloc[:,4:5] #1 = none, 0 = completed

race = base2.iloc[:,1:2]
race = OHE.fit_transform(race).toarray() #columns = ["A", "B", "C", "D", "E"]

edu = base2.iloc[:,2:3]
edu = OHE.fit_transform(edu).toarray()#columns=["Associate","B.S.","HS","M.A.","Some College","Some HS"]

raceDf = pd.DataFrame(data = race, index = range(1000), columns = ["A", "B", "C", "D", "E"])
eduDf=pd.DataFrame(data=edu,index=range(1000),columns=["Associate","B.S.","HS","M.A.","Some College","Some HS"])

df1 = pd.concat([gender, raceDf, eduDf, lunch, course], axis=1)
df1 = pd.concat([df1, data_math], axis=1)

x_train, x_test, y_train, y_test = train_test_split(df1.iloc[:,:-1], df1.iloc[:,-1:], test_size = 0.33, random_state = 0)

regressor.fit(x_train, y_train)
y_pred = regressor.predict(x_test)
y_pred = pd.DataFrame(data = y_pred, index=range(330), columns = ["Y PRED"])

X_l = np.append(arr = np.ones((1000, 1)).astype(int), values = df1, axis = 1)

X_l = df1.iloc[:,[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]].values
X_l = np.array(X_l, dtype = float)
model = sm.OLS(df1.iloc[:,-1:], X_l).fit()
print(model.summary())

X_l = df1.iloc[:,[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14]].values
X_l = np.array(X_l, dtype = float)
model = sm.OLS(df1.iloc[:,-1:], X_l).fit()
print(model.summary())

X_l = df1.iloc[:,[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 14]].values
X_l = np.array(X_l, dtype = float)
model = sm.OLS(df1.iloc[:,-1:], X_l).fit()
print(model.summary())

x_train = x_train.iloc[:,1:]
x_test = x_test.iloc[:,1:]

regressor.fit(x_train, y_train)
y_pred1 = regressor.predict(x_test)
y_pred1 = pd.DataFrame(data = y_pred1, index=range(330), columns = ["Y PRED 1"])

X_l = df1.iloc[:,[0, 1, 2, 4, 5, 6, 7, 8, 9, 10, 12, 14]].values
X_l = np.array(X_l, dtype = float)
model = sm.OLS(df1.iloc[:,-1:], X_l).fit()
print(model.summary())

X_l = df1.iloc[:,[0, 1, 2, 4, 5, 6, 7, 8, 9, 12, 14]].values
X_l = np.array(X_l, dtype = float)
model = sm.OLS(df1.iloc[:,-1:], X_l).fit()
print(model.summary())

X_l = df1.iloc[:,[0, 1, 2, 5, 6, 7, 8, 9, 12, 14]].values
X_l = np.array(X_l, dtype = float)
model = sm.OLS(df1.iloc[:,-1:], X_l).fit()
print(model.summary())

x_train = x_train.iloc[:,1:]
x_test = x_test.iloc[:,1:]

regressor.fit(x_train, y_train)
y_pred2 = regressor.predict(x_test)
y_pred2 = pd.DataFrame(data = y_pred2, index=range(330), columns = ["Y PRED 2"])

y_testValues = y_test.values
y_testDf = pd.DataFrame(data = y_testValues, index = range(330), columns = ["Math Score"])
corrDf = pd.concat([y_pred, y_pred1, y_pred2, y_testDf], axis=1)

