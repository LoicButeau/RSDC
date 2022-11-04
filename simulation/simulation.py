from turtle import pd
import yfinance as yf
import pandas as pd
import pandas_ta as ta
import matplotlib.pyplot as plt
from backtesting import Backtest, Strategy


class Cta(Strategy):
    def init(self):
        super().init()
        #self.atr = self.I(atr, pd.Series(self.data.High),pd.Series(self.data.Low),pd.Series(self.data.Close))
        #self.macd =self.I(macd, pd.Series(self.data.Close))
        self.donchian20H=self.I(donchian20H,pd.Series(self.data.High),pd.Series(self.data.Low))
        self.donchian20L=self.I(donchian20L,pd.Series(self.data.High),pd.Series(self.data.Low))
        self.donchian10L = self.I(donchian10L, pd.Series(self.data.High),pd.Series(self.data.Low))
    
    def next(self):

        price = self.data.Close[-1]

        if (price == self.donchian20H[-1]):
            self.buy()
        if(price == self.donchian10L[-1]):
            self.position.close()



def atr(H,L,C):
    atr = ta.atr(H,L,C)
    return atr.to_numpy()

def macd(C):
    macd = ta.macd(C)
    return macd["MACDh_12_26_9"].to_numpy()

def donchian20H(H,L):
    donch = ta.donchian(H,L)
    return donch["DCU_20_20"].to_numpy()

def donchian20L(H,L):
    donch = ta.donchian(H,L)
    return donch["DCL_20_20"].to_numpy()

def donchian10L(H,L):
    donch2 = ta.donchian(H,L,lower_length=10)
    return donch2["DCL_10_20"].to_numpy()


data = yf.download("ZC=F", start="2022-6-30", end="2022-10-30", interval = "1h")


bt = Backtest(data,Cta, cash=10_000)
stats=bt.run()
bt.plot()

print(stats)



#x_axis = range(len(data))
#y_axis = data["Close"]
#y2_axis=donch["DCL_20_20"]
#y3_axis=donch["DCU_20_20"]

#plt.plot(x_axis,y_axis)
#plt.plot(x_axis, y2_axis)
#plt.plot(x_axis,y3_axis)
#plt.show()

