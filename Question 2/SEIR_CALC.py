import numpy as np
from scipy.integrate import odeint
import pandas as pd

# Defined parameters and initial conditions
annual_growth_rate = 0.01
alpha = 0.13040167618273908
beta = 0.12936643303985582
gamma = 0.5213003472739447
delta = 1.4210526315789473
A0 = 372.88
H0 = 2057
X0 = 373.2
N = 477408

# SEIR model differential equations including population growth
def d_states_dt(states, t, N, alpha, beta, gamma, delta, growth_rate):
    A, H, X = states
    new_population = growth_rate * N
    dA_dt = -alpha * A + delta * X - gamma * A + new_population
    dH_dt = alpha * A - beta * H
    dX_dt = beta * H - delta * X
    return [dA_dt, max(dH_dt, 0), dX_dt]  # Ensure never negative

# Time settings for the simulation over 50 years
years_to_simulate = 50
t = np.linspace(0, years_to_simulate, int(years_to_simulate) + 1)

# Solve the differential equations
states0 = [A0, H0, X0]
states = odeint(d_states_dt, states0, t, args=(N, alpha, beta, gamma, delta, annual_growth_rate))

# Extract the Homeless results
Homeless = states[:, 1]

# Prepare the data for the DataFrame
data = {
    'Year': np.arange(2008, 2008+years_to_simulate+1).tolist(),
    'Homeless': Homeless.tolist()
}

# Create the DataFrame
homeless_df = pd.DataFrame(data)

# Save the DataFrame to a CSV file
homeless_df.to_csv('homeless_projection.csv', index=False)

# Print success message
print('Homeless data for the next 50 years has been saved to homeless_projection.csv')