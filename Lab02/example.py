#1.

import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

lambda1 = 4

lambda2 = 6

probabilitate_primul_mecanic = 0.4

numar_clienti = 10000

valori_X = []
for _ in range(numar_clienti):
    probabilitate = np.random.rand()
    if probabilitate < probabilitate_primul_mecanic:
        valori_X.append(stats.expon(scale=1/lambda1).rvs())
    else:
        valori_X.append(stats.expon(scale=1/lambda2).rvs())

media_X = np.mean(valori_X)
deviatia_standard_X = np.std(valori_X)

print(f"Media lui X: {media_X}")
print(f"Deviatia standard a lui X: {deviatia_standard_X}")

plt.hist(valori_X, bins=50, density=True, alpha=0.6, color='g', label='Densitate')
plt.xlabel('X (Timp de servire)')
plt.ylabel('Densitate')
plt.title('Distributia lui X')
plt.legend(loc='best')

x = np.linspace(0, max(valori_X), 100)
y1 = stats.expon(scale=1/lambda1).pdf(x)
y2 = stats.expon(scale=1/lambda2).pdf(x)
plt.plot(x, y1, 'r-', lw=2, label=f'Expon({lambda1})')
plt.plot(x, y2, 'b-', lw=2, label=f'Expon({lambda2})')

plt.legend()
plt.show()


#2.

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gamma

params = [(4, 3), (4, 2), (5, 2), (5, 3)]

probabilities = [0.25, 0.25, 0.30, 0.20]

x_values = np.linspace(0, 20, 1000)

total_density = np.zeros_like(x_values)
for i, (alpha, beta) in enumerate(params):
    server_density = gamma.pdf(x_values, alpha, scale=1/beta)
    total_density += server_density * probabilities[i]

plt.figure(figsize=(10, 6))
plt.plot(x_values, total_density, label='Densitatea lui X')
plt.xlabel('Timp (milisecunde)')
plt.ylabel('Densitate de probabilitate')
plt.title('Densitatea distributiei lui X')
plt.legend()
plt.grid(True)
plt.show()


#3.




