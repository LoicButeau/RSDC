import numpy as np
import matplotlib.pyplot as plt
import math
import scipy.optimize as spo



delta1 =0.3
delta2=0.2
phi=0.5
sig=1

transmat= [[0.5,0.5],
           [0.5,0.5]]


xk = np.arange(len(transmat))

def generate_sample(cur_state):
    return np.random.choice(xk, 1,  p = transmat[cur_state])

initial_state = 0
sample_len = 100
y = [-1 for i in range(sample_len)]
y[0] = initial_state
for i in range(1, sample_len):
    y[i] = generate_sample(y[i-1])[0]



p11=transmat[0][0]
p12=1-p11
p21=transmat[1][0]
p22=1-p21
params= [delta1,delta2,phi,sig,p11,p12,p21,p22]

def constraint1(params):
    return 1-params[4] - params[5]

def constraint2(params):
    return 1-params[6]-params[7]

def constraint3(params):
    return params[3]

def hamilton(params):
    p1filt = np.zeros(len(y))
    p2filt = np.zeros(len(y))
    f= np.zeros(len(y))
    
    p1filt[0] = ((1 - params[7]) / (2 - params[4] - params[7]))
    p2filt[0] = ((1 - params[4]) / (2 - params[4] - params[7]))

    print(params)
    #initialisation des params[4]filt
    for t in range(1,len(y)):
    
    
        p1pred = params[4]*p1filt[t-1] + params[6]*p2filt[t-1]
        p2pred = params[5]*p1filt[t-1] + params[7]*p2filt[t-1]
    
        f[t] = (1/(math.sqrt(2*np.pi*params[3])*math.exp(-(y[t]-params[0]-params[2]*y[t-1])**2/(2*params[3])))*p1pred + 1/(math.sqrt(2*np.pi*params[3])*math.exp(-(y[t]-params[1]-params[2]*y[t-1])**2/(2*params[3])))*p2pred)
    
        p1filt[t] = 1/(math.sqrt(2*np.pi*params[3])*math.exp(-(y[t]-params[0]-params[2]*y[t-1])**2/(2*params[3])))*p1pred / f[t]
        p2filt[t] = 1/(math.sqrt(2*np.pi*params[3])*math.exp(-(y[t]-params[1]-params[2]*y[t-1])**2/(2*params[3])))*p2pred / f[t]

    log = np.sum(f)
    return -log

con1 = {"type": "eq", "fun": constraint1}
con2 = {"type": "eq", "fun": constraint2}
con3 = {"type": "ineq", "fun": constraint3}
cons = [con1,con2,con3]
bounds = ((None,None),(None,None),(None,None),(0.01,1),(0,1),(0,1),(0,1),(0,1))
xo=[0.1,0.1,0.1,0.1,0.5,0.5,0.5,0.5]


optparams = spo.minimize(hamilton, x0=xo,constraints=cons, bounds = bounds)
print(optparams.x)

x_axis= range(len(y))
y_axis=y

plt.plot(x_axis, y_axis)
plt.show()