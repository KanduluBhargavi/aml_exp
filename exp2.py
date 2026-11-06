import numpy as np

def gradient_descent(x,y):
  m_curr=b_curr=0
  rate=0.1
  n=len(x)
  for i in range(10):
    y_predicted=m_curr*x+b_curr
    cost=(1/n)* sum([val**2 for val in (y-y_predicted)])
    md=-(2/n)*sum(x*(y-y_predicted))
    bd=-(2/n)*sum(y-y_predicted)
    m_curr=m_curr-rate*md
    b_curr=b_curr-rate*bd
    print("m{}, b{}, cost{} iteration{}".format(m_curr,b_curr,cost,i))
x=np.array([1,2,3])
y=np.array([2,4,5])
gradient_descent(x,y)