#This will determine whether we use a poisson model or a negative binomial regression
import pandas as pd

# Load your dataset
df = pd.read_csv('dataset.csv')

# Calculate the observed mean and variance of the count data
mean_homeless = df['TotalHomeless'].mean()
variance_homeless = df['TotalHomeless'].var()

# Calculate the dispersion statistic (variance-to-mean ratio)
dispersion_statistic = variance_homeless / mean_homeless
print(f"Dispersion Statistic: {dispersion_statistic}")

# Check for overdispersion
if dispersion_statistic > 1:
    print("The count data is overdispersed.")
else:
    print("The count data is not overdispersed.")