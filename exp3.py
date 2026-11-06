import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression,Ridge,Lasso
from sklearn.metrics import mean_squared_error
#x1 is important, x2 is less important and x3 is noise
X=np.array([
    [1,2,10],[2,1,12],[3,3,9],[4,5,11],[5,4,13],[6,6,8],[7,7,7],[8,8,14]
])
y=np.array([6,7,12,18,20,24,28,32])
df=pd.DataFrame(X,columns=["X1","X2","X3"])
df["y"]=y
print("dataset",df)
#without regularization
lr=LinearRegression()
lr.fit(X,y)
y_pred_lr=lr.predict(X)

print("no regularization")
print(":coefficients",lr.coef_)
print("intercept",lr.intercept_)
print("MSE",mean_squared_error(y,y_pred_lr))
#ridge regression
ridge=Ridge(alpha=1.0)
ridge.fit(X,y)
y_pred_ridge=ridge.predict(X)

print("ridge regression")
print("Coefficients",ridge.coef_)
print("intercept",ridge.intercept_)
print("MSE",mean_squared_error(y,y_pred_ridge))
#LASSO regularization
lasso=Lasso(alpha=0.5)
lasso.fit(X,y)
y_pred_lasso=lasso.predict(X)
print("lasso regularization")
print("coefficients",lasso.coef_)
print("intercept",lasso.intercept_)
print("MSE",mean_squared_error(y,y_pred_lasso))
#second dataset X2,X1 ARE USEFUL BUT X3 IS NOISE HAS NO RELATION WITH Y
X=np.array([
    [1,1,10],[2,2,30],[3,3,80],[4,4,30],[5,5,60],[6,6,10],[7,7,90],[8,8,40]
])
#Y=3*X1+2*X2

y=np.array([5,10,15,20,25,30,35,40])
print("X\n",X)
print("y\n ",y)
#normal regression
lr=LinearRegression()
lr.fit(X,y)
y_pred_lr=lr.predict(X)

print("no regularization")
print("coefficients",lr.coef_)
 #Ridge regression

ridge=Ridge(alpha=1.0)
ridge.fit(X,y)
print("ridge")
print("coefficient",ridge.coef_)

 #lasso
lasso=Lasso(alpha=0.5)
print("lasso")
lasso.fit(X,y)
print("coefficients",lasso.coef_)