import numpy as np
import matplotlib.pyplot as plt
import math
import scipy.optimize as spo
from hmmlearn.hmm import GaussianHMM
import random


transmat= [[0.5,0.5],
           [0.5,0.5]]
delta1 =0.3
delta2=0.2
phi=0.5
sig=0.3
p11=transmat[0][0]
p12=1-p11
p21=transmat[1][0]
p22=1-p21
params= [delta1,delta2,phi,sig,p11,p12,p21,p22]




# Generate the AR(1) observations
def generateAR():

    st1 = []
    st2 = []
    timeseries = []
    T=100
    delta1 = 0.3
    delta2=0.1
    phi=0.2
    et = np.random.normal(0, 1, T)

    st1.append(delta1/(1-phi))
    st2.append(delta2/(1-phi))

    for i in range(1,T):
        st1.append(delta1 + phi*st1[i-1]+et[i])
        st2.append(delta2 + phi*st2[i-1]+et[i])
    
    st1.extend(st2)
    random.shuffle(st1)   
    return st1

y = generateAR()

#use the package to infer the transtition matrix to validate our results with the hamilton filter
model = GaussianHMM(n_components = 2)
X = np.array(y).reshape(-1,1)
model.fit(X)
print(model.transmat_)

#Constraints for the optimization of parameters
def constraint1(params):
    return 1-params[4] - params[5]

def constraint2(params):
    return 1-params[6]-params[7]

def constraint3(params):
    return params[3]-0.00001

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

    log = np.log(np.sum(f))
    return -log

#Optimization
con1 = {"type": "eq", "fun": constraint1}
con2 = {"type": "eq", "fun": constraint2}
con3 = {"type": "ineq", "fun": constraint3}
cons = [con1,con2,con3]
bounds = ((None,None),(None,None),(None,None),(0.0001,1),(0,1),(0,1),(0,1),(0,1))
xo=[0.1,0.1,0.1,0.1,0.2,0.8,0.3,0.7]
optparams = spo.minimize(hamilton, x0=xo,constraints=cons, bounds = bounds)
print(optparams.x)

#Plotting the graph
x_axis= range(len(y))
y_axis=y

plt.plot(x_axis, y_axis)
plt.show()