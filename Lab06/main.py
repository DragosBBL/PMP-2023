import pymc as pm
import arviz as az
import matplotlib.pyplot as plt

def create_poisson_binomial_model(y, theta):
    with pm.Model() as model:
        # Prior distribution for the number of trials
        n = pm.Poisson('n', mu=10)
        # Likelihood of the observed data
        y_obs = pm.Binomial('y_obs', n=n, p=theta, observed=y)
        # Sampling from the posterior distribution
        trace = pm.sample(1000, return_inferencedata=True)
    return trace

# Dictionary to store results
results = {}

# Define combinations of y and theta values
combinations = [(y, theta) for y in [0, 5, 10] for theta in [0.2, 0.5]]

# Iterate through combinations and store results in the dictionary
for y, theta in combinations:
    trace = create_poisson_binomial_model(y, theta)
    results[(y, theta)] = trace

# Plotting the posterior distributions
fig, axes = plt.subplots(len(combinations) // 2, 2, figsize=(10, 8), squeeze=False)

for i, ((y, theta), trace) in enumerate(results.items()):
    ax = axes[i // 2, i % 2]
    az.plot_posterior(trace, var_names=['n'], ax=ax)
    ax.set_title(f'Y={y}, Theta={theta}')

plt.tight_layout()
plt.show()
