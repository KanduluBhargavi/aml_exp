import numpy as np
from scipy.stats import f
samsung=np.array([36000,35000,31000,42000,38000])
google=np.array([44000,45000,51000,50000,52000])
sony=np.array([16000,15000,18000,24000,22000])
mean_samsung=np.mean(samsung)
mean_google=np.mean(google)
mean_sony=np.mean(sony)

print("mean samsung",mean_samsung)
print("mean google",mean_google)
print("mean sony",mean_sony)
all_data=np.concatenate([samsung,google,sony])
grand_mean=np.mean(all_data)
print("grand mean",grand_mean)
n_samsung=len(samsung)
n_google=len(google)
n_sony=len(sony)

SSB =(n_samsung*(mean_samsung-grand_mean)**2+ n_google*(mean_google-grand_mean)**2+n_sony*(mean_sony-grand_mean)**2)
print("SSB=",SSB)
SSW_samsung=np.sum((samsung-mean_samsung)**2)
SSW_google=np.sum((google-mean_google)**2)
SSW_sony=np.sum((sony-mean_sony)**2)
SSW=SSW_samsung+SSW_google+SSW_sony

print("SSW_samsung",SSW_samsung)
print("SSW_google",SSW_google)
print("SSW_sony",SSW_sony)
print("SSW",SSW)
k=3#no of groups
N=len(all_data) #no of observation
df_between=k-1
df_within=N-k
df_total=N-1

print("df_between",df_between)
print("df_within",df_within)
print("df_total",df_total)
MSB=SSB/df_between
MSW=SSW/df_within
print("MS BETWEEN",MSB)
print("MS Within",MSW)
F_calculated=MSB/MSW
print("F_calculated",F_calculated)
alpha=0.05
F_critical=f.ppf(1-alpha,df_between,df_within)
print("alpha",alpha)
print("F_critical",F_critical)
if F_calculated>F_critical:
    print("Reject H0")
    print("There is a significant difference among the group means")
else:
   print("FAil to reject H0")
   print("There is no significant difference among the group means")
