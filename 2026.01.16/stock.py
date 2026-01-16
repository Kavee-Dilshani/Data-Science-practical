import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from sklearn.linear_model import LinearRegression


#read dataset
df_stock = pd.read_csv('./AAPL.csv')
df_stock.info()

df_stock = df_stock.rename(columns = {'Close(t)' : 'Close'})
print(df_stock.head())
print(df_stock.tail(5))
print(df_stock.shape)
print(df_stock.columns)
df_stock = df_stock.drop(columns = 'Date')
df_stock = df_stock.drop(columns = 'Date_col')

#plot
# df_stock['Close'].plot(figsize = (10,7))
# plt.title("Stock Price", fontsize = 17)
# plt.ylabel('Price', fontsize = 14)
# plt.xlabel('Time', fontsize = 14)
# plt.grid(which = "major", color = 'k', linestyle = '-.', linewidth = 0.5)
# plt.show()

def splitTrainTest(df, p = 0.8):
     y = df['Close_forcast']
     df = df.drop(columns = 'Close_forcast')
     print(df.shape)
     n = df.shape[0]
     ntr = int(n*p)
     
     x_Train, y_Train = df[:ntr], y[:ntr]
     x_Test, y_Test = df[ntr:], y[ntr:]
     print('Total - ',n)
     print('Training - ',x_Train.shape,y_Train.shape)
     print('Test - ',x_Test.shape, y_Test.shape  )

     return x_Train,y_Train,x_Test, y_Test

# Xtr, Ytr, Xte, Yte = splitTrainTest(df_stock)
# lr = LinearRegression()
# lr.fit(X.train,y_Train)
# print('LR_Coefficieants: \n',lr.coef_)
# print('Intercept: \n',lr.intercept_)

#read dataset
df_stock = pd.read_csv('./AAPL.csv')
df_stock = df_stock.drop(columns = 'Date')
df_stock = df_stock.drop(columns = 'Date_col')

Xtr, Ytr, Xte, Yte = splitTrainTest(df_stock)
lr = LinearRegression()
lr.fit(X.train,y_Train)
print('LR_Coefficieants: \n',lr.coef_)
print('Intercept: \n',lr.intercept_)

#testing
Y_train_pred = lr.predict(Xtr)
X_Train_pred =  lr.predict(Ytr)

print('Training R-Squared: ', round(metrices.r2_score(Ytr, Y_train_pred),2))
print('Teating R-Squared: ', round(metrices.r2_score(Yte, Y_test_pred),2))

df_pred = pd.DataFrame(Yte.values, columns = ['Actual'])
df_pred['Predicted']  = Y_test_pred
df_pred[['Actual', 'Predicted']].plot()
plt.show()

