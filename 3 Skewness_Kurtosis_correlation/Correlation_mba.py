import pandas as pd

mba = pd.read_csv("D:\\HimaniSingh\\LIvewire\\machine,learning\\dataset\\mba.csv")
mba=mba.iloc[1:10,1:3]
mba.gmat.corr(mba.workex)#negative correlation

#Graphs
x=mba['gmat']
y=mba['workex']
import matplotlib.pyplot as plt
plt.plot(x,y)
