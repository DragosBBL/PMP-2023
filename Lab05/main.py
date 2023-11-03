import pymc as pm
import pandas as pd
import arviz as az
import matplotlib.pyplot as plt


data = pd.read_csv('trafic.csv')

minute = data['minut'].tolist()
numar_masini = data['nr. masini'].values

ore_crestere = [7, 16]
ore_descrestere = [8, 19]

with pm.Model() as model:
    # Define a prior distribution for the Poisson rate parameter 'lambda_poisson'
    lambda_poisson = pm.Normal('lambda_poisson', mu=10, sigma=5)

    # Create the observed Poisson distribution 'trafic' with the rate parameter 'lambda_poisson'
    trafic = pm.Poisson('trafic', mu=lambda_poisson, observed=numar_masini)

    # Modify the rate parameter 'lambda_poisson' based on the specified hours of increase and decrease
    for hour in ore_crestere:
        lambda_poisson = pm.Deterministic(f'lambda_poisson_increase{hour}', lambda_poisson * 1.2)
    for hour in ore_descrestere:
        lambda_poisson = pm.Deterministic(f'lambda_poisson_decrease{hour}', lambda_poisson * 0.8)

# Perform Bayesian sampling to estimate the posterior distribution
with model:
    trace = pm.sample(2000)

# Convert the trace to a pandas DataFrame for further analysis
df = trace.to_dataframe(trace)

# Create a posterior plot using ArviZ
az.plot_posterior(trace)

# Display the posterior plot
plt.show()