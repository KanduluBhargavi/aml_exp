import numpy as np
from scipy.stats import chi2
observed=np.array([[45,5],[15,35]])
print("OBSERVED FREQUENCY TABLE")
print("-"*50)

print("                pass       fail")
print("high attendance",observed[0])
print("low attendance",observed[1])
row_totals=observed.sum(axis=1)
print("\n ROW TOTALS")
print("-"*50)
print("high attendance",row_totals[0])
print("low attendance",row_totals[1])
column_totals=observed.sum(axis=0)
print("\n COLUMN TOTALS")
print("-"*50)
print("pass",column_totals[0])
print("fail",column_totals[1])
grand_total=observed.sum()
print("\n GRAND TOTAL")
print("-"*50)
print(grand_total)
expected=np.zeros_like(observed,dtype=float)
for i in range(observed.shape[0]):
  for j in range(observed.shape[1]):
    expected[i][j]=(row_totals[i]*column_totals[j])/grand_total
print(expected)
difference=observed-expected
print(difference)
chi_squared_components=(difference**2)/expected
print(chi_squared_components)
chi_square=chi_squared_components.sum()
print(chi_square)
rows,columns=observed.shape
df=(rows-1)*(columns-1)
print("degrees of freedom")
print("rows",rows)
print("columns",columns)
print("df",df)
alpha=0.5
critical_value=chi2.ppf(
    1-alpha,df
)
print("alpha",alpha)
print("critical value")
print(critical_value)
if chi_square>critical_value:
  print("Reject H0")
  print("attendance and result are significantly associated")
else:
  print("fail to reject H0")
  print("attendance and result are independent")