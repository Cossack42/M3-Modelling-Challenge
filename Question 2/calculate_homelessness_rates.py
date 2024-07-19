import pandas as pd

# Read the data from the CSV file
homelessness_data = pd.read_csv('homelessness_data_bnh.csv')

# Prepare lists to hold calculated values for gamma and delta
gammas = [None]  # Initial None for the first delta value
deltas = [None]  # Initial None for the first gamma value

# Calculate gamma (Improvement rate) and delta (Relapse rate)
for i in range(1, len(homelessness_data)):
    # Calculate gamma
    change_eligible_not_homeless = (homelessness_data.loc[i, 'Eligible_Not_Homeless'] -
                                    homelessness_data.loc[i - 1, 'Eligible_Not_Homeless'])
    gamma = -change_eligible_not_homeless / homelessness_data.loc[i - 1, 'Eligible_Not_Homeless']
    gammas.append(gamma)

    # Calculate delta - Note we use the assumption that an increase in priority need might indicate relapse
    change_priority_need = (homelessness_data.loc[i, 'Priority_Need'] -
                            homelessness_data.loc[i - 1, 'Priority_Need'])
    if change_priority_need > 0 and change_eligible_not_homeless < 0:
        delta = change_priority_need / (homelessness_data.loc[i - 1, 'Priority_Need'] -
                                        homelessness_data.loc[i - 1, 'Eligible_Not_Homeless'])
        deltas.append(delta)
    else:
        deltas.append(None)  # Cannot calculate delta if we don't have a decrease in 'Eligible_Not_Homeless'

# Add the calculated values back to the DataFrame
homelessness_data['Gamma'] = gammas
homelessness_data['Delta'] = deltas

# Optional: fill NaN for first missing values of gamma and delta
homelessness_data.fillna(value={'Gamma': 0, 'Delta': 0}, inplace=True)

# Print the DataFrame with the calculated values
print(homelessness_data)

# Calculate the average Gamma and Delta, ignoring 'None' values and preventing division by zero
average_gamma = sum(g for g in gammas if g is not None) / (len([g for g in gammas if g is not None]) or 1)
filtered_deltas = [d for d in deltas if d is not None]
average_delta = sum(filtered_deltas) / (len(filtered_deltas) or 1)


#Display average Gamma and Delta values
print(f"Average Gamma: {average_gamma}")
print(f"Average Delta: {average_delta}")

