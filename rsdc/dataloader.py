import pandas as pd
import matplotlib.pyplot as plt
import math
import numpy as np
import hmmnorm as hmm

insample = pd.read_excel("In_Sample_Pelletier.xlsx")

#Column names are GBP/USD DM/USD JPY/USD CHF/USD


log = np.log(insample)
logdiff = log.diff()
logdiff = logdiff.iloc[1:]

figure, axis = plt.subplots(4, 1)
x= range(len(logdiff["GBP/USD"]))
axis[0].plot(x,logdiff["GBP/USD"])
axis[1].plot(x,logdiff["DM/USD"])
axis[2].plot(x,logdiff["JPY/USD"])
axis[3].plot(x,logdiff["CHF/USD"])





u = np.mean(logdiff["GBP/USD"])
sig = np.std(logdiff["GBP/USD"])
df = pd.DataFrame(columns = ["U0", "U1", "sig", "p11", "p22"])

GBPoptim = hmm.optimize(logdiff["GBP/USD"])
GBPparams = GBPoptim.x

DMoptim = hmm.optimize(logdiff["DM/USD"])
DMparams = DMoptim.x

JPYoptim = hmm.optimize(logdiff["JPY/USD"])
JPYparams = JPYoptim.x

CHFoptim = hmm.optimize(logdiff["CHF/USD"])
CHFparams = CHFoptim.x

df.loc[len(df)]=GBPparams
df.loc[len(df)]=DMparams
df.loc[len(df)]=JPYparams
df.loc[len(df)]=CHFparams


print(df)


plt.show()


