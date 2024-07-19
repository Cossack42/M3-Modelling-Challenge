import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.formula.api import glm
from statsmodels.genmod.families import NegativeBinomial
from functools import reduce
import matplotlib.pyplot as plt

# Load data from separate CSV files; ensure they all have a 'Year' column and the same length
housing_units_df = pd.read_csv('housing_units_data.csv')
sales_price_df = pd.read_csv('sales_price_data.csv')
population_df = pd.read_csv('population_data.csv')
income_df = pd.read_csv('income_data.csv')
homelessness_df = pd.read_csv('homeless_projection.csv')

# Merge all dataframes on the 'Year' column
df_merged = reduce(
    lambda left, right: pd.merge(left, right, on=['Year']),
    [housing_units_df, sales_price_df, population_df, income_df, homelessness_df]
)

# Handle any missing values if necessary
df_merged = df_merged.dropna()

# Fit a Negative Binomial regression model; estimate alpha using a preliminary model or specify it directly
# Define the regression formula, ensure column names match those in the merged dataframe exactly
formula = 'TotalHomeless ~ TotalHousingUnits + MedianSalesPrice + TotalPopulation + MedianIncome'

# Initial Poisson model to estimate alpha
poisson_model = glm(formula, data=df_merged, family=sm.families.Poisson()).fit()
alpha_est = poisson_model.pearson_chi2 / poisson_model.df_resid

# Negative Binomial model
nb_model = glm(formula, data=df_merged, family=NegativeBinomial(alpha=alpha_est)).fit()

# Print the model summary
print(nb_model.summary())

# Visualization of coefficients
coefs = pd.DataFrame({
    'coef': nb_model.params.values[1:],  # Excludes Intercept
    'err': nb_model.bse.values[1:],  # Excludes Intercept
    'varname': nb_model.params.index[1:]  # Excludes Intercept
})

fig, ax = plt.subplots(figsize=(10, 6))
coefs.plot(x='varname', y='coef', kind='bar', ax=ax, yerr='err', capsize=4)
plt.title('Coefficient Estimates From Negative Binomial Regression')
plt.ylabel('Coefficients')
plt.axhline(y=0, color='black', linestyle='--')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
