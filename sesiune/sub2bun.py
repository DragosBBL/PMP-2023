import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# nr aruncari
n = 5

# simulam prima stema folosind dis geom
p_geom = 1.0 / (n + 1)  # param pt dis geom
geom_distribution = stats.geom.pmf(np.arange(1, n + 1), p_geom)
h = 3
t = 10

x = np.linspace(0, 1, 100)

# true posterior pt beta
true_posterior = stats.beta.pdf(x, h + 1, t + 1)

# quadratic approx
mean_q = {'p': (h + 1) / (h + t + 2)}
std_q = np.sqrt(mean_q['p'] * (1 - mean_q['p']) / (h + t + 2))  # deviatia standard
quadratic_approximation = stats.norm.pdf(x, mean_q['p'], std_q)

plt.plot(x, true_posterior, label='True posterior')
plt.plot(x, quadratic_approximation, label='Quadratic approximation')

# dis geom pt prima stema
plt.stem(np.arange(1, n + 1) / (n + 1), geom_distribution, basefmt=" ", linefmt='--', markerfmt='o', label='Geometric(0)')

plt.legend(loc=0, fontsize=13)
plt.title(f'heads = {h}, tails = {t}')
plt.xlabel('θ', fontsize=14)
plt.yticks([])

plt.show()


#THETA E 3/13 (h+t/t)