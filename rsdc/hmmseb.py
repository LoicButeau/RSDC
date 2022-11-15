import numpy as np
import matplotlib.pyplot as plt
import math
import scipy.optimize as spo
from hmmlearn.hmm import GaussianHMM
import random
import pandas as pd
import numpy.random as rnd

np.random.seed(833)

#Generate a random time series depending of the transition matrix
# p11 = 0.95
# p12 = 1 - p11
# p22 = 0.9
# p21 = 1- p22
# mat_transition = np.array([[p11,p12],[p21,p22]])

# #generate two time series
# delta1 =3
# delta2=0.2
# phi=0.1
# sig=1
# params= [delta1,delta2,phi,sig,p11,p22]
# n=300

# Generates the states
def generate_state(mat_transition, n):
    x=np.arange(n)
    x[0]=1
    dim=len(mat_transition[0])
    for i in np.arange(1,n):
        x[i]=rnd.choice(a=dim,p=mat_transition[x[i-1]])
    return x

# Generate thes AR
def generateAR(params, n,x):
    
    st = []
    T=n
    delta1 = params[0]
    delta2= params[1]
    phi=params[2]
    et = np.random.normal(0, 1, T)

    st1 = delta1/(1-phi)

    #Since we start in state 1, we add st1
    st.append(st1)

    for i in range(1,T):

        if (x[i] == 1):
            st.append(delta1 + phi*st[i-1]+et[i])
        else:
            st.append(delta2 + phi*st[i-1]+et[i])

    return pd.DataFrame({"series_f": st, "state": x})


#Generate one time series with two time series and the transition matrix
#def generate_series(mat_transition, params, n):
    
    
    #series_1 is use to get the length of one series
    serie_total = generateAR(params, n)
    
    series_1 = serie_total.iloc[:,0] 
    series_2 =serie_total.iloc[:,1] 
    
    #Initialize our parameters
    series_f = np.zeros(len(series_1))

    
    
    state = generate_state(mat_transition, len(series_1))
    
    for i in range(len(series_f)):
        series_f[i] = serie_total.iloc[i,state[i]]
        
    return pd.DataFrame({"series_f":series_f, "series_1": series_1, "series_2": series_2, "state": state    })
    

#generates probility
def generate_p(y, params) :
    y=y["series_f"] #temporaire pour faciliter la fonction
    p1filt = np.zeros(len(y))
    p2filt = np.zeros(len(y))
    p1_prev = np.zeros(len(y))
    p2_prev = np.zeros(len(y))
    fyt_global = np.zeros(len(y))

    delta1 = params[0]
    delta2= params[1]
    phi = params[2]
    sig=params[3]
    p11 = params[4]
    p22=params[5]
    p21 = 1 - p22
    p12 = 1 - p11


    #we initialize
    p1filt[0] = ((1 - p22) / (2 - p11 - p22))
    p2filt[0] = ((1 - p11) / (2 - p11 - p22))


    for t in range(1, len(y)) : 
        #etape 1
        p1_prev[t] = p1filt[t-1] * p11 + p2filt[t-1] * p21
        p2_prev[t] = p1filt[t-1] * p12 + p2filt[t-1] * p22
        
        
        fyt_global[t] = 1/(math.sqrt(2*np.pi*sig))*math.exp(-1*(y[t]-delta1-phi*y[t-1])**2/(2*sig))*p1_prev[t] + 1/(math.sqrt(2*np.pi*sig))*math.exp(-1*(y[t]-delta2-phi*y[t-1])**2/(2*sig))*p2_prev[t]
    
        p1filt[t] = (1/(math.sqrt(2*np.pi*sig)))*math.exp(-1*(y[t]-delta1-phi*y[t-1])**2/(2*sig))* p1_prev[t] / fyt_global[t]
        p2filt[t] = (1/(math.sqrt(2*np.pi*sig)))*math.exp(-1*(y[t]-delta2-phi*y[t-1])**2/(2*sig))* p2_prev[t] / fyt_global[t]

    return pd.DataFrame({"fyt":fyt_global, "p1filt": p1filt,"p1prev": p1_prev, "p2filt": p2filt, "P2prev": p2_prev  })


#Function to create HMM des probabilité fil trouvé préalablement
def hmm(result, y) :
    y=y["series_f"]
    hmm = np.zeros(len(y))
    for i in range(1, len(y)):
        if result["p1filt"][i] > result["p2filt"][i]:
            hmm[i] = 0
        elif result["p1filt"][i] == result["p2filt"][i]:
            hmm[i]= 0.5
        else: hmm[i] = 1
    return hmm



def find_transition(result, y):
    s = hmm(result, y)
    mt11 = 0
    mt12 = 0
    mt21 = 0
    mt22 = 0
    for m in range(len(y)-1):
        
        
        if s[m] == 0 :
            if s[m+1]  == 0:
                mt11 = mt11 + 1
            else : 
                mt12 =  mt12 + 1
        else:
            if s[m+1]  == 0:
                mt21 = mt21 + 1
            else : 
                mt22 = mt22 + 1
    
    sum_all1 = mt11 + mt12 
    sum_all2 = mt21 + mt22  
    
    matrice_transition_estime = np.array([[(mt11/sum_all1),(mt12/sum_all1) ], 
            [(mt21/sum_all2), (mt22/sum_all2)]])
    return matrice_transition_estime
    

def hamilton(params,y):
    yt=y["series_f"]
    p1filt = np.zeros(len(yt))
    p2filt = np.zeros(len(yt))
    f= np.zeros(len(yt))

    delta1 = params[0]
    delta2= params[1]
    phi = params[2]
    sig=params[3]
    p11 = params[4]
    p12 = 1 - p11
    p22=params[5]
    p21 = 1 - p22

    
    p1filt[0] = ((1 - p22) / (2 - p11 - p22))
    p2filt[0] = ((1 - p11) / (2 - p11 - p22))

    for t in range(1,len(yt)):

        #etape 1
        p1pred = p11*p1filt[t-1] + p21*p2filt[t-1]
        p2pred = p12*p1filt[t-1] + p22*p2filt[t-1]
    
        #etape 2
        f[t] = 1/(math.sqrt(2*np.pi*sig))*math.exp(-1*(yt[t]-delta1-phi*yt[t-1])**2/(2*sig))*p1pred + 1/(math.sqrt(2*np.pi*sig))*math.exp(-1*(yt[t]-delta2-phi*yt[t-1])**2/(2*sig))*p2pred
            
        #etape 3
        p1filt[t] = 1/(math.sqrt(2*np.pi*sig))*math.exp(-1*(yt[t]-delta1-phi*yt[t-1])**2/(2*sig))*p1pred / f[t]
        p2filt[t] = 1/(math.sqrt(2*np.pi*sig))*math.exp(-1*(yt[t]-delta2-phi*yt[t-1])**2/(2*sig))*p2pred / f[t]

    res = f[1:]
    log = np.sum((np.log(res)))
    #print(np.log(res))
    return -1*log

def optimize(y):
    #Constraints for the optimization of parameters
    
    def constraint1(params):
        return 1-params[4] - (1-params[4])

    def constraint2(params):
        return 1-params[5]-(1-params[5])

    def constraint3(params):
        return params[3]-0.00001

    def constraint4(params):
        return 1-abs(params[2])

    #Optimization
    con1 = {"type": "eq", "fun": constraint1}
    con2 = {"type": "eq", "fun": constraint2}
    con3 = {"type": "ineq", "fun": constraint3}
    con4 = {"type": "ineq" , "fun" : constraint4}
    cons = [con1,con2,con3,con4]
    bounds = ((None,None),(None,None),(None,None),(0.0001,1),(0,1),(0,1))
    xo=[0,0,0,0.01,0.01,0.01]
    optparams = spo.minimize(hamilton,args=y, x0=xo,constraints=cons, bounds = bounds)
    return optparams


def MonteCarloSimulate(count):
    
    for i in range(count):
        p11= random.random()+0.0001
        p22 = random.random() + 0.0001
        p12 = 1 - p11
        p21 = 1 - p22
        delta1 = random.uniform(0,4) + 0.0001
        delta2 = random.uniform(0,4) + 0.0001
        phi = random.uniform(0,2) + 0.0001
        sig=1
        t=200

        transmat = [[p11,p12],[p21,p22]]
        params=[delta1,delta2,phi,sig,p11,p22]

        x = generate_state(transmat, t)
        y = generateAR(params,t,x)
        result = generate_p(y, params)

        transition = (find_transition(result,y))
        opttransition = optimize(y)

        print(params)
        print(transition)
        print(opttransition.message)


    
MonteCarloSimulate(4)





