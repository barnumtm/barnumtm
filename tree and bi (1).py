#!/usr/bin/env python
# coding: utf-8

# In[165]:


import yfinance as yf      #import yahoo data
import numpy as np         #importing numpy
import pandas as pd        ~importing pandas library 
import matplotlib as mpl
import os
import matplotlib.pyplot as plt


# In[166]:


initial_data = yf.download("MAR", start="2020-10-01", end="2021-10-31") #downloading MAR data


# In[167]:


data =  initial_data['Adj Close'] # renaming data with adj close date


# In[4]:


data.plot(figsize=(10, 12), subplots=True) #plot with closing price over time 


# In[5]:


data.describe().round(2) # describing the data  collected from yfinance 


# In[149]:


S0 = 151.12           # spot stock price
K = 154.00             # strike
T = 1.0                 # maturity 
r = .0143               # risk free rate 10 Years US govt Treasury 
sig = 0.21              # volatility
N = 5                   # number of periods or number of time steps  
payoff = "put"          # payoff 


# In[150]:


dT = float(T) / N                             # Delta t
u = np.exp(sig * np.sqrt(dT))                 # up factor
d = 1.0 / u                                   # down factor 


# In[151]:


S = np.zeros((N + 1, N + 1))
S[0, 0] = S0
z = 1
for t in range(1, N + 1):
    for i in range(z):
        S[i, t] = S[i, t-1] * u
        S[i+1, t] = S[i, t-1] * d
    z += 1


# In[152]:


S


# In[153]:


a = np.exp(r * dT)    # risk free compound return
p = (a - d)/ (u - d)  # risk neutral up probability
q = 1.0 - p           # risk neutral down probability
p


# In[154]:


S_T = S[:,-1]
V = np.zeros((N + 1, N + 1))
if payoff =="call":
    V[:,-1] = np.maximum(S_T-K, 0.0)
elif payoff =="put":
    V[:,-1] = np.maximum(K-S_T, 0.0)
V


# In[155]:


# for European Option
for j in range(N-1, -1, -1):
    for i in range(j+1):
        V[i,j] = np.exp(-r*dT) * (p * V[i,j + 1] + q * V[i + 1,j + 1])
V


# In[156]:


print('European ' + payoff, str( V[0,0]))


# In[133]:


def mcs_simulation_np(p):
    M = p
    I = p
    dt = T / M 
    S = np.zeros((M + 1, I))
    S[0] = S0 
    rn = np.random.standard_normal(S.shape) 
    for t in range(1, M + 1): 
        S[t] = S[t-1] * np.exp((r - sigma ** 2 / 2) * dt + sigma * np.sqrt(dt) * rn[t]) 
    return S


# In[144]:


T = 1.0                       #Time
r = 0.0143                    # 10 Years US govt Treasury 
sigma = 0.21                  # volatility
S0 = 151.12                   # Spot AS OF 2/12/2021
K = 154.00                    #k is strike


# In[145]:


S = mcs_simulation_np(1000)     #run the simulation 1000 


# In[146]:


S = np.transpose(S)
S


# In[147]:


n, bins, patches = plt.hist(x=S[:,-1], bins='auto', color='#0504aa',alpha=0.7, rwidth=0.85)

plt.grid(axis='y', alpha=0.75)
plt.xlabel('S_T')
plt.ylabel('Frequency')
plt.title('Frequency distribution of the simulated end-of-preiod values')


# In[148]:


p = np.mean(np.maximum(K - S[:,-1],0))
print('European put', str(p))


# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as si


# In[ ]:


delta(15112, 154.00, 1.00, 0.0143, 0.00, 0.21, 'put')


# In[ ]:


#S: spot price
#K: strike price
#T: time to maturity
#r: risk free rate
#q: continuous dividend yield
#vol: volatility of underlying asset
#payoff: call or put


# In[ ]:


delta(151.12, 154.00, 1.00, 0.0143, 0.00, 0.21, 'call') # list


# In[ ]:


delta(151.12, 154.00, 1.00, 0.0143, 0.00, 0.21, 'put')

# Greeks 


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as si


# In[4]:


#S: spot price
#K: strike price
#T: time to maturity
#r: risk free rate
#q: continuous dividend yield
#vol: volatility of underlying asset
#payoff: call or put


# In[5]:


def delta(S, K, T, r, q, vol, payoff):
    
    d1 = (np.log(S / K) + (r - q + 0.5 * vol ** 2) * T) / (vol * np.sqrt(T))
    if payoff == "call":
        delta = np.exp(-q * T) * si.norm.cdf(d1, 0.0, 1.0)
    elif payoff == "put":
        delta =  - np.exp(-q * T) * si.norm.cdf(-d1, 0.0, 1.0)
    
    return delta


# In[6]:


delta(151.12, 154.00, 1.00, 0.0143, 0.00, 0.21, 'call')


# In[7]:


delta(151.12, 154.00, 1.00, 0.0143, 0.00, 0.21, 'put')


# In[13]:


S = np.linspace(50,150,11)
Delta_Call = np.zeros((len(S),1))
Delta_Put = np.zeros((len(S),1))
for i in range(len(S)):
    Delta_Call [i] = delta(S[i], 151.12, 1.00, 0.0143, 0.00, 0.21, 'call')
    Delta_Put [i] = delta(S[i], 151.12, 1.00, 0.0143, 0.00, 0.21, 'put')


# In[14]:


fig = plt.figure()
plt.plot(S, Delta_Call, '-')
plt.plot(S, Delta_Put, '--')
plt.grid()
plt.xlabel('Stock Price')
plt.ylabel('Delta')
plt.title('Delta')
plt.legend(['Delta for Call','Delta for Put'])


# In[15]:


def gamma(S, K, T, r, q, vol, payoff):
    
    d1 = (np.log(S / K) + (r - q + 0.5 * vol ** 2) * T) / (vol * np.sqrt(T))
    gamma = np.exp(- q * T) * si.norm.pdf(d1, 0.0, 1.0) / (vol * S * np.sqrt(T))
    
    return gamma


# In[16]:


gamma(151.12, 154.00, 1.00, 0.0143, 0.00, 0.21, 'call')


# In[17]:


gamma(151.12, 154.00, 1.00, 0.0143, 0.00, 0.21, 'put')


# In[18]:


S = np.linspace(50,150,11)
Gamma = np.zeros((len(S),1))
for i in range(len(S)):
    Gamma [i] = gamma(S[i], 151.12, 1.00, 0.0143, 0.00, 0.21, 'call')


# In[19]:


fig = plt.figure()
plt.plot(S, Gamma, '-')
plt.grid()
plt.xlabel('Stock Price')
plt.ylabel('Gamma')
plt.title('Gamma')
plt.legend(['Gamma for Call and Put'])


# In[20]:


def speed(S, K, T, r, q, vol, payoff):
    
    d1 = (np.log(S / K) + (r - q + 0.5 * vol ** 2) * T) / (vol * np.sqrt(T))
    speed = - np.exp(-q * T) * si.norm.pdf(d1, 0.0, 1.0) / ((vol **2) * (S**2) * np.sqrt(T)) * (d1 + vol * np.sqrt(T))
    
    return speed


# In[21]:


speed(151.12, 154.00, 1.00, 0.0143, 0.00, 0.21, 'call')


# In[22]:


speed(151.12, 154.00, 1.00, 0.0143, 0.00, 0.21, 'put')


# In[23]:


S = np.linspace(50,150,11)
Speed = np.zeros((len(S),1))
for i in range(len(S)):
    Speed [i] = speed(S[i], 151.12, 1.00, 0.0143, 0.00, 0.21, 'call')


# In[24]:


fig = plt.figure()
plt.plot(S, Speed, '-')
plt.grid()
plt.xlabel('Stock Price')
plt.ylabel('Speed')
plt.title('Speed')
plt.legend(['Speed for Call and Put'])


# In[27]:


def theta(S, K, T, r, q, vol, payoff):
    
    d1 = (np.log(S / K) + (r - q + 0.5 * vol ** 2) * T) / (vol * np.sqrt(T))
    d2 = (np.log(S / K) + (r - q - 0.5 * vol ** 2) * T) / (vol * np.sqrt(T))
    if payoff == "call":
        theta = vol * S * np.exp(-q * T) * si.norm.pdf(d1, 0.0, 1.0) / (2 * np.sqrt(T)) - q * S * np.exp(-q * T) * si.norm.cdf(d1, 0.0, 1.0) + r * K * np.exp(-r * T) * si.norm.cdf(d2, 0.0, 1.0)
    elif payoff == "put":
        theta = vol * S * np.exp(-q * T) * si.norm.pdf(-d1, 0.0, 1.0) / (2 * np.sqrt(T)) - q * S * np.exp(-q * T) * si.norm.cdf(-d1, 0.0, 1.0) + r * K * np.exp(-r * T) * si.norm.cdf(-d2, 0.0, 1.0)
    
    return theta


# In[28]:


theta(151.12, 154.00, 1.00, 0.0143, 0.00, 0.21, 'call')


# In[29]:


theta(151.12, 154.00, 1.00, 0.0143, 0.00, 0.21, 'put')


# In[30]:


T = np.linspace(0.25,3,12)
Theta_Call = np.zeros((len(T),1))
Theta_Put = np.zeros((len(T),1))
for i in range(len(T)):
    Theta_Call [i] = theta(100, 100, T[i], 0.05, 0.03, 0.25, 'call')
    Theta_Put [i] = theta(100, 100, T[i], 0.05, 0.03, 0.25, 'put')


# In[31]:


fig = plt.figure()
plt.plot(T, Theta_Call, '-')
plt.plot(T, Theta_Put, '-')
plt.grid()
plt.xlabel('Time to Expiry')
plt.ylabel('Theta')
plt.title('Theta')
plt.legend(['Theta for Call', 'Theta for Put'])


# In[32]:


def rho(S, K, T, r, q, vol, payoff):
    
    d1 = (np.log(S / K) + (r - q + 0.5 * vol ** 2) * T) / (vol * np.sqrt(T))
    d2 = (np.log(S / K) + (r - q - 0.5 * vol ** 2) * T) / (vol * np.sqrt(T))
    if payoff == "call":
        rho =  K * T * np.exp(-r * T) * si.norm.cdf(d2, 0.0, 1.0)
    elif payoff == "put":
        rho = - K * T * np.exp(-r * T) * si.norm.cdf(-d2, 0.0, 1.0)
    
    return rho


# In[33]:


rho(151.12, 154.00, 1.00, 0.0143, 0.00, 0.21, 'call')


# In[34]:


rho(151.12, 154.00, 1.00, 0.0143, 0.00, 0.21, 'put')


# In[35]:


r = np.linspace(0,0.1,11)
Rho_Call = np.zeros((len(r),1))
Rho_Put = np.zeros((len(r),1))
for i in range(len(r)):
    Rho_Call [i] = rho(100, 100, 1, r[i], 0.03, 0.25, 'call')
    Rho_Put [i] = rho(100, 100, 1, r[i], 0.03, 0.25, 'put')


# In[36]:


fig = plt.figure()
plt.plot(r, Rho_Call, '-')
plt.plot(r, Rho_Put, '-')
plt.grid()
plt.xlabel('Interest Rate')
plt.ylabel('Rho')
plt.title('Rho')
plt.legend(['Rho for Call', 'Rho for Put'])



