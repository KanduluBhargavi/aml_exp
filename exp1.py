import numpy as np
X1=np.array([3,1,2,4,3])
X2=np.array([6,3,4,7,5])
y=np.array([1,0,0,1,1])
X=np.column_stack((np.ones(5),X1,X2))
B=np.linalg.inv(X.T @ X) @ X.T @ y
b0,b1,b2=B
print("intercept(b0)",b0)
print("coefficient for studied hours(b1)",b1)
print("coefficient  for sleep hours(b2)", b2)
import math
def sigmoid(x):
  return 1/(1+math.exp(-x))
z=X@B
print("linear combination(z)")
print(z)
#to get probabilites
y_prob=np.array([sigmoid(val) for val in z])
print("predicted values",y_prob)
y_pred=(y_prob>=0.5).astype(int)
print("predicted values",y_pred)
TP=np.sum((y==1) & (y_pred==1))
TN=np.sum((y==0) & (y_pred==0))
FP=np.sum((y==0) & (y_pred==1))
FN=np.sum((y==1) & (y_pred==0))

print("TP",TP,"FP",FP)
print("FN",FN,"TN",TN)
accuracy=(TP+TN) /len(y_pred)
precision=TP/(TP+FP) if (TP+FP)!=0 else 0
recall=TP/(TP+FN) if (TP+FN)!=0 else 0
f1score=2*(precision * recall)/(precision+recall) if (precision+recall)!=0 else 0
print(accuracy,precision, recall, f1score)