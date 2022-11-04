import numpy as np
import matplotlib.pyplot as plt
import math
import scipy.optimize as spo
from hmmlearn.hmm import GaussianHMM
import random
import pandas as pd

np.random.seed(145)

transmat= [[0.5,0.5],
           [0.5,0.5]]
delta1 =0.3
delta2=0.1
phi=0.2
sig=0.3
p11=transmat[0][0]
p12=1-p11
p21=transmat[1][0]
p22=1-p21





# Generate the AR(1) observations
def generateAR():

    st1 = []
    st2 = []
    timeseries = []
    T=100
    delta1 = 3
    delta2=0.1
    phi=0.2
    et1 = np.random.normal(0, 1, T)
    et2 = np.random.normal(0,1,T)


    st1.append([delta1/(1-phi),0])
    st2.append([delta2/(1-phi),1])

    for i in range(1,T):
        st1.append([delta1 + phi*st1[i-1][0]+et1[i],0])
        st2.append([delta2 + phi*st2[i-1][0]+et2[i],1])
    

    st1.extend(st2)
    random.shuffle(st1)
    st1=np.array(st1).T       
    return st1[0], st1[1]

y,state = generateAR()


#Constraints for the optimization of parameters
def constraint1(params):
    return 1-params[4] - params[5]

def constraint2(params):
    return 1-params[6]-params[7]

def constraint3(params):
    return params[3]-0.00001

def constraint4(params):
    return 1-abs(params[2])


params= [delta1,delta2,phi,sig,p11,p12,p21,p22]

#Hamilton filter algorithm
def hamilton(params):
    p1filt = np.zeros(len(y))
    p2filt = np.zeros(len(y))
    f= np.zeros(len(y))
    
    p1filt[0] = ((1 - params[7]) / (2 - params[4] - params[7]))
    p2filt[0] = ((1 - params[4]) / (2 - params[4] - params[7]))

    for t in range(1,len(y)):

        #etape 1
        p1pred = params[4]*p1filt[t-1] + params[6]*p2filt[t-1]
        p2pred = params[5]*p1filt[t-1] + params[7]*p2filt[t-1]
    
        #etape 2
        f[t] = 1/(math.sqrt(2*np.pi*params[3]))*math.exp(-1*(y[t]-params[0]-params[2]*y[t-1])**2/(2*params[3]))*p1pred + 1/(math.sqrt(2*np.pi*params[3]))*math.exp(-1*(y[t]-params[1]-params[2]*y[t-1])**2/(2*params[3]))*p2pred
            
        #etape 3
        p1filt[t] = 1/(math.sqrt(2*np.pi*params[3]))*math.exp(-1*(y[t]-params[0]-params[2]*y[t-1])**2/(2*params[3]))*p1pred / f[t]
        p2filt[t] = 1/(math.sqrt(2*np.pi*params[3]))*math.exp(-1*(y[t]-params[1]-params[2]*y[t-1])**2/(2*params[3]))*p2pred / f[t]

    res = f[1:]
    log = np.sum((np.log(res)))
    #print(np.log(res))
    return -1*log

def hamiltonfilt(params):
    p1filt = np.zeros(len(y))
    p2filt = np.zeros(len(y))
    f= np.zeros(len(y))
    
    p1filt[0] = ((1 - params[7]) / (2 - params[4] - params[7]))
    p2filt[0] = ((1 - params[4]) / (2 - params[4] - params[7]))

    for t in range(1,len(y)):

        #etape 1
        p1pred = params[4]*p1filt[t-1] + params[6]*p2filt[t-1]
        p2pred = params[5]*p1filt[t-1] + params[7]*p2filt[t-1]
    
        #etape 2
        f[t] = 1/(math.sqrt(2*np.pi*params[3]))*math.exp(-1*(y[t]-params[0]-params[2]*y[t-1])**2/(2*params[3]))*p1pred + 1/(math.sqrt(2*np.pi*params[3]))*math.exp(-1*(y[t]-params[1]-params[2]*y[t-1])**2/(2*params[3]))*p2pred
            
        #etape 3
        p1filt[t] = 1/(math.sqrt(2*np.pi*params[3]))*math.exp(-1*(y[t]-params[0]-params[2]*y[t-1])**2/(2*params[3]))*p1pred / f[t]
        p2filt[t] = 1/(math.sqrt(2*np.pi*params[3]))*math.exp(-1*(y[t]-params[1]-params[2]*y[t-1])**2/(2*params[3]))*p2pred / f[t]

    return p1filt,p2filt


#Optimization
con1 = {"type": "eq", "fun": constraint1}
con2 = {"type": "eq", "fun": constraint2}
con3 = {"type": "ineq", "fun": constraint3}
con4 = {"type": "ineq" , "fun" : constraint4}
cons = [con1,con2,con3,con4]
bounds = ((None,None),(None,None),(None,None),(0.0001,1),(0,1),(0,1),(0,1),(0,1))
xo=[0.1,0.1,0.1,0.1,0.5,0.5,0.5,0.5]
optparams = spo.minimize(hamilton, x0=xo,constraints=cons, bounds = bounds)
print(optparams.x)

p1filt,p2filt = hamiltonfilt(optparams.x)

y3_axis=np.zeros(len(y))
i=0
for p in p1filt:
    if (p > 0.5):
        y3_axis[i] = 0
    else:
        y3_axis[i] = 1
    i += 1



#Plotting the graph
x_axis= range(len(y))
y_axis=y
y2_axis = state


plt.plot(x_axis, y_axis)
plt.plot(x_axis,y2_axis)
plt.plot(x_axis,y3_axis)
plt.show()
