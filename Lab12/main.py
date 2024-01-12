import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

#ex1
def posterior_grid(grid_points=10, heads=6, tails=9, prior_type='uniform'):
    grid = np.linspace(0, 1, grid_points)

    if prior_type == 'uniform':
        prior = np.repeat(1 / grid_points, grid_points)
    elif prior_type == 'binary':
        prior = (grid <= 0.5).astype(int)
    elif prior_type == 'absolute_difference':
        prior = abs(grid - 0.5)
    else:
        prior = np.ones(grid_points)

    likelihood = stats.binom.pmf(heads, heads + tails, grid)
    posterior = likelihood * prior
    posterior /= posterior.sum()

    return grid, posterior


data = np.repeat([0, 1], (10, 3))
points = 10
h = data.sum()
t = len(data) - h
prior_types = ['uniform', 'binary', 'absolute_difference']

# Plotting
plt.figure(figsize=(15, 5))
for i, prior_type in enumerate(prior_types, 1):
    grid, posterior = posterior_grid(points, h, t, prior_type)
    plt.subplot(1, len(prior_types), i)
    plt.plot(grid, posterior, 'o-')
    plt.title(f'Prior: {prior_type.capitalize()} | Heads = {h}, Tails = {t}')
    plt.yticks([])
    plt.xlabel('θ')

plt.tight_layout()
plt.show()

#ex2

def estimate_pi(N):
    x, y = np.random.uniform(-1, 1, size=(2, N))
    inside = (x**2 + y**2) <= 1
    pi = inside.sum() * 4 / N
    error = abs((pi - np.pi) / pi) * 100
    return error

N_values = [100, 1000, 10000]

num_trials = 100

errors = np.zeros((len(N_values), num_trials))

for i, N in enumerate(N_values):
    for j in range(num_trials):
        errors[i, j] = estimate_pi(N)

mean_errors = np.mean(errors, axis=1)
std_dev_errors = np.std(errors, axis=1)

plt.errorbar(N_values, mean_errors, yerr=std_dev_errors, fmt='o-', capsize=5)
plt.xscale('log')
plt.xlabel('Number of Points (N)')
plt.ylabel('Error (%)')
plt.title('Estimation of pi with Error Bars')
plt.show()

#3



