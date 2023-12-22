import numpy as np
import arviz as az
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture

clusters = 3
n_cluster = [200, 150, 150]
n_total = sum(n_cluster)
means = [5, 0, -5]
std_devs = [2, 2, 2]

mix = np.random.normal(np.repeat(means, n_cluster), np.repeat(std_devs, n_cluster))

az.plot_kde(np.array(mix))

plt.show()

for n_components in range(2, 5):
    model = GaussianMixture(n_components=n_components, random_state=42)
    model.fit(mix.reshape(-1, 1))

    x = np.linspace(min(mix), max(mix), 1000)
    y = np.exp(model.score_samples(x.reshape(-1, 1)))

    plt.plot(x, y, label=f'{n_components} componente')

plt.title('Model de mixtură de distribuţii Gaussiene pe datele furnizate')
plt.legend()
plt.show()