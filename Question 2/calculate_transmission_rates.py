import pandas as pd

# Available Houses Data for calculating beta
available_houses_data = pd.read_csv('available_houses_bnh.csv')
# Total Eligible Households Data for calculating alpha
homelessness_data = pd.read_csv('homelessness_data_bnh.csv')

# Initialize lists for alpha and beta calculations
alpha_list = [None]  # No alpha for the first year
beta_list = [None]  # No beta for the first year

# Calculate alpha and beta together since they depend on consecutive years of data
for i in range(1, len(homelessness_data)):
    # Alpha Calculation: Year over year change in total eligible households for homelessness
    increase_in_eligible_households = (homelessness_data.loc[i, 'Total_Eligible_Households'] -
                                       homelessness_data.loc[i-1, 'Total_Eligible_Households'])
    # Avoid division by zero by ensuring the previous value is at least 1
    alpha = abs(increase_in_eligible_households) / max(homelessness_data.loc[i-1, 'Total_Eligible_Households'], 1)
    alpha_list.append(alpha)

    # Beta Calculation: Change in the number of available houses year-to-year
    # Assuming available housing units data aligns with total eligible households data
    if i < len(available_houses_data):
        # Absolute value ensures that both increases and decreases in availability affect beta
        change_in_houses_available = (available_houses_data.loc[i, 'Houses_Available'] -
                                      available_houses_data.loc[i - 1, 'Houses_Available'])
        beta = abs(change_in_houses_available) / max(available_houses_data.loc[i - 1, 'Houses_Available'], 1)
        beta_list.append(beta)
    else:
        # If there isn't corresponding data in housing units, append None
        beta_list.append(None)

# Update the homelessness DataFrame with the calculated alpha and beta values
homelessness_data['Alpha'] = alpha_list
# Align the length of beta_list with homelessness_data in case of length mismatch
# This can happen if the housing units data has fewer records
while len(beta_list) < len(homelessness_data):
    beta_list.append(None)
homelessness_data['Beta'] = beta_list

# Calculate and print the average alpha and beta, excluding 'None' values
average_alpha = sum(filter(None, alpha_list)) / len([a for a in alpha_list if a is not None])
average_beta = sum(filter(None, beta_list)) / len([b for b in beta_list if b is not None])

# Display the DataFrame with the calculated Alpha and Beta values
print(homelessness_data[['Year', 'Alpha', 'Beta']])

#Display average Alpha and Beta values
print(f"Average Alpha: {average_alpha}")
print(f"Average Beta: {average_beta}")