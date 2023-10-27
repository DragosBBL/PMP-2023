import numpy as np

# 1. Definiția modelului probabilistic
lambda1 = 20  # Rata de sosire a clienților (Poisson) în clienți/oră
medie_comanda = 2  # Media timpului de plasare și plată a comenzii în minute
deviatie_comanda = 0.5  # Deviația standard a timpului de plasare și plată a comenzii

# 2. Determinarea valorii maxime a lui alpha pentru o probabilitate de 95% și un timp maxim de 15 minute
probabilitate = 0.95
t_maxim = 15
alpha_max = -t_maxim / np.log(1 - probabilitate)

print("Valoarea maximă a lui alpha:", alpha_max)

# 3. Calcularea timpului mediu de așteptare pentru a fi servit unui client
alpha_max = 1 / alpha_max  # Reciproca valorii alpha_max
medie_comanda = medie_comanda + 1 / alpha_max  # Timpul mediu de așteptare la comandă
print("Timpul mediu de așteptare:", medie_comanda, "minute")

# Simularea timpului total de servire pentru mai mulți clienți
num_samples = 1000  # Numărul de simulări
timp_pregatire = np.random.exponential(scale=1 / alpha_max, size=num_samples)
timp_total = medie_comanda + timp_pregatire

# Exemplu de calcul al timpului mediu de așteptare pentru toți clienții
timp_mediu_total = np.mean(timp_total)
print("Timpul mediu total de așteptare pentru toți clienții:", timp_mediu_total, "minute")
