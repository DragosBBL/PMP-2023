import pymc3 as pm
import numpy as np
import matplotlib.pyplot as plt

# Generam 200 de timpi medii de asteptare de o distributie normala
np.random.seed(42)  # Pentru reproducibilitate
timpi_medii_asteptare = np.random.normal(loc=10, scale=2, size=200)

# Definim modelul in PyMC
with pm.Model() as model:
    # Alegem distributii a priori pentru parametrii medie (u) și dispersie (o)
    u = pm.Normal('u', mu=10, sd=2)
    sigma = pm.HalfNormal('sigma', sd=2)

    # Definim distributia normala a datelor observate
    observed_data = pm.Normal('observed_data', mu=u, sd=sigma, observed=timpi_medii_asteptare)

# Efectuam inferenta Bayesiană
if __name__ == '__main__':
    with model:
        trace = pm.sample(1000, tune=1000)

    # Vizualizam distributia a posteriori pentru parametrul o
    pm.plot_posterior(trace['sigma'], hdi_prob=0.95)
    plt.title('Distribuția a posteriori pentru o')
    plt.xlabel('Valoare a lui o')
    plt.ylabel('Densitatea de probabilitate')
    plt.show()


#Modelul: se defineste un model in care timpul mediu de asteptare este modelat printr-o distributie normala, cu parametrii mediei (u) și dispersiei (o).
#Distributia normala a datelor observate este definita pentru a reflecta variabilitatea observatiilor
#u medie : Distributie medie cu medie 10 și dispersie (sd) 2
#o dispersie : Distributie HalfNormal cu dispersie 2
#Distributiile a priori sunt alese pentru a reflecta cunostintele sau presupunerile initiale despre parametrii modelului intainte de a observa datele
#Alegerea distributiilor normale este comuna pentru parametrii care sunt susceptibili de a avea o distributie gaussiana in practica



