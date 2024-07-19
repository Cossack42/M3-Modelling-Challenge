import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import odeint
from scipy.stats import linregress

# Annual population growth rate for Brighton and Hove
annual_growth_rate = 0.01

# Use the calculated average values for transition rates
alpha = 0.015496127233052227  # Transmission rate from At-Risk to Homeless
beta = 0.18429413982985354  # Transmission rate from Homeless to Housed
gamma = 0.03619756660739407 # Recovery rate of At-Risk individuals
delta = 0.14592595905172612  # Relapse rate from Housed to At-Risk

# Initial populations in each compartment
A0 = 403.56  # Initial At-Risk population
H0 = 2057    # Initial Homeless population
X0 = 403.56  # Initial Housed (formerly homeless) population
N = 262019   # Total population

# SEIR model differential equations including population growth
def d_states_dt(states, t, N, alpha, beta, gamma, delta, growth_rate):
    A, H, X = states
    new_population = growth_rate * N  # New people added to the population annually
    dA_dt = -alpha * A + delta * X - gamma * A + new_population  # New at-risk
    dH_dt = alpha * A - beta * H  # New homeless
    dX_dt = beta * H - delta * X  # New housed

    # Prevent negative populations
    dA_dt = max(dA_dt, -A)
    dH_dt = max(dH_dt, -H)
    dX_dt = max(dX_dt, -X)

    return [dA_dt, dH_dt, dX_dt]

# Time settings for the simulation over 50 years
years_to_simulate = 50
t = np.linspace(0, years_to_simulate, years_to_simulate + 1)  # One entry per year

# Solve the differential equations
states0 = [A0, H0, X0]
states = odeint(d_states_dt, states0, t, args=(N, alpha, beta, gamma, delta, annual_growth_rate))

# Extract the results
At_Risk, Homeless, Housed = states.T

# Specific years we want to extract data for
years_of_interest = [1, 2, 7, 8, 10, 11, 20, 30, 50]
data_points = {year: {} for year in years_of_interest}

# Extract data points for specific years
for year in years_of_interest:
    index = int(year)
    data_points[year]["At_Risk"] = At_Risk[index]
    data_points[year]["Homeless"] = Homeless[index]
    data_points[year]["Housed"] = Housed[index]

# Print the extracted data
for year, data in data_points.items():
    print(f"Year {year}: {data}")

# Plot the results
plt.figure(figsize=(12, 8))
plt.plot(t, At_Risk, label='At-Risk (A)', color='orange')
plt.plot(t, Homeless, label='Homeless (H)', color='red')
plt.plot(t, Housed, label='Housed (X)', color='green')
plt.title('Adapted SEIR Model Dynamics with Population Growth (Brighton and Hove)')
plt.xlabel('Time (years)')
plt.ylabel('Population')
plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), shadow=True, ncol=3)

plt.tight_layout()  # Adjust the padding to make room for the legend
plt.grid(True)
plt.show()

# Hypothetical observed data points for the Homeless population (for specific years)
observed_H_data = np.array([1755, 1916, 2487])
# Corresponding prediction years in the simulation
prediction_years = [1, 2, 7]  # These are indexes in your simulation that correspond to the observed data points
# Make sure this matches the actual observed data

# Extract the predicted Homeless values for the same specific years:
seir_H_predictions = states[:, 1]  # Assuming column 1 corresponds to the Homeless compartment 'H'
predicted_H_data = np.array([seir_H_predictions[year] for year in prediction_years])

# Verify observed and predicted arrays have the same length
assert len(observed_H_data) == len(predicted_H_data), "The length of observed and predicted data arrays must match."

# Calculate R-squared using linear regression from SciPy:
slope, intercept, r_value, p_value, std_err = linregress(observed_H_data, predicted_H_data)
r_squared = r_value**2

print(f"The R-squared value for the Homeless compartment is: {r_squared}")