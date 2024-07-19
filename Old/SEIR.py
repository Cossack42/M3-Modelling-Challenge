import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

# SEIR model differential equations with time-varying beta
def seir_model(y, t, N, beta_func, sigma, gamma):
    S, E, I, R = y
    beta = beta_func(t)
    dSdt = -beta * I * S / N
    dEdt = beta * I * S / N - sigma * E
    dIdt = sigma * E - gamma * I
    dRdt = gamma * I
    return dSdt, dEdt, dIdt, dRdt

# Set up initial conditions and parameters
N = 15000000  # total population
I0 = 225000    # initial number of infected individuals
E0 = 0    # initial number of exposed individuals
S0 = N - I0 - E0  # initial number of susceptible individuals
R0 = 0    # initial number of recovered individuals
sigma = 0.00032  # exposure rate
gamma = 0.0009 # recovery rate

# Define the time-varying beta function
def beta_func(t):
    return (-0.1921 * (t/365)**2 + 3.21 * (t/365)) / 100

# Set up the time grid (e.g., days)
t = np.arange(0, 6000, 1)

# Solve the SEIR equations with time-varying beta
solution = odeint(seir_model, [S0, E0, I0, R0], t, args=(N, beta_func, sigma, gamma))

# Extract results
S, E, I, R = solution.T

# Plot the results
plt.plot(t, S, label='Susceptible')
plt.plot(t, E, label='Exposed')
plt.plot(t, I, label='Infectious')
plt.plot(t, R, label='Recovered')
plt.xlabel('Days')
plt.ylabel('Population')
plt.legend()
plt.show()