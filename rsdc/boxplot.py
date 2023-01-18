import matplotlib.pyplot as plt
import pandas as pd

df1 = pd.read_csv("result1.csv")
df2= pd.read_csv("result2.csv")
df5 = pd.read_csv("result5.csv")


del df1[df1.columns[0]]


figure, axis = plt.subplots(3, 5)
axis[0,0].boxplot(df1["u0e"])
axis[0,0].set_title("Uo")

axis[0,1].boxplot(df1["u1e"])
axis[0,1].set_title("u1")

axis[0,2].boxplot(df1["sige"])
axis[0,2].set_title("sig")

axis[0,3].boxplot(df1["p11e"])
axis[0,3].set_title("p11")

axis[0,4].boxplot(df1["p22e"])
axis[0,4].set_title("p22")


axis[1,0].boxplot(df2["u0e"])
axis[1,1].boxplot(df2["u1e"])
axis[1,2].boxplot(df2["sige"])
axis[1,3].boxplot(df2["p11e"])
axis[1,4].boxplot(df2["p22e"])

axis[2,0].boxplot(df5["u0e"])
axis[2,1].boxplot(df5["u1e"])
axis[2,2].boxplot(df5["sige"])
axis[2,3].boxplot(df5["p11e"])
axis[2,4].boxplot(df5["p22e"])


plt.show()
