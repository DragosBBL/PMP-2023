import pymc3 as pm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 1.
# incarcam datele
data = pd.read_csv('Admission.csv')
data['GRE_normalized'] = (data['GRE'] - data['GRE'].mean()) / data['GRE'].std()
data['GPA_normalized'] = (data['GPA'] - data['GPA'].mean()) / data['GPA'].std()

if __name__ == '__main__':
    # construim modelul logistic folosind pymc3
    with pm.Model() as logistic_model:
        beta0 = pm.Normal('beta0', mu=0, sd=10)
        beta1 = pm.Normal('beta1', mu=0, sd=10)
        beta2 = pm.Normal('beta2', mu=0, sd=10)

        # definim formula logistica
        logit_p = beta0 + beta1 * data['GRE_normalized'] + beta2 * data['GPA_normalized']

        # adaguam distributia bernoulli pentru admitere
        admission = pm.Bernoulli('admission', logit_p=logit_p, observed=data['Admission'])

        # esantionam din distributia a posteriori folosind pymc3
        trace = pm.sample(1000, tune=1000)

    # 2.
    # esantionam val pentru beta0 beta1 si beta2
    beta0_samples = trace['beta0'][:, np.newaxis]
    beta1_samples = trace['beta1'][:, np.newaxis]
    beta2_samples = trace['beta2'][:, np.newaxis]

    # definim functia logistica inversa
    def logistic_inv(x, beta0, beta1, beta2):
        return 1 / (1 + np.exp(-(beta0 + beta1 * x[0] + beta2 * x[1])))

    # generam gridul pt a reprezenta zona de decizie
    gre_values = np.linspace(data['GRE_normalized'].min(), data['GRE_normalized'].max(), 100)
    gpa_values = np.linspace(data['GPA_normalized'].min(), data['GPA_normalized'].max(), 100)
    grid = np.array(np.meshgrid(gre_values, gpa_values)).T.reshape(-1, 2)

    # calc prob si reprezentam zona de decizie
    probabilities = np.mean(np.exp(logistic_inv(grid.T, beta0_samples, beta1_samples, beta2_samples)), axis=0)
    probabilities_matrix = probabilities.reshape(len(gre_values), len(gpa_values))

    plt.figure(figsize=(10, 8))
    plt.scatter(data['GRE_normalized'], data['GPA_normalized'], c=data['Admission'], cmap='viridis', alpha=0.8)
    contour = plt.contour(gre_values, gpa_values, probabilities_matrix, levels=[0.5], colors='red')
    plt.clabel(contour, inline=True, fontsize=12, fmt='Decision Boundary')

    plt.xlabel('GRE Normalized')
    plt.ylabel('GPA Normalized')
    plt.title('Logistic Regression Decision Boundary')
    plt.show()

    # 3.
    # datele pt noul student
    new_student_data = {'GRE_normalized': (550 - data['GRE'].mean()) / data['GRE'].std(),
                        'GPA_normalized': (3.5 - data['GPA'].mean()) / data['GPA'].std()}

    # esantionam din distributia a posteriori pt prob de admitere pentru noul student
    with logistic_model:
        post_pred = pm.sample_posterior_predictive(trace, samples=5000, var_names=['admission'], model=logistic_model)

    # calculam prob medii si intervalul de 90% HDI
    admission_probabilities = np.mean(post_pred['admission'], axis=0)
    prob_interval_90 = np.percentile(admission_probabilities, [5, 95])

    print(f'Intervalul de 90% HDI pentru probabilitatea de admitere: {prob_interval_90}')

    # 4.
    # datele pentru al doilea student
    new_student_data_2 = {'GRE_normalized': (500 - data['GRE'].mean()) / data['GRE'].std(),
                          'GPA_normalized': (3.2 - data['GPA'].mean()) / data['GPA'].std()}

    # esantionam din distributia a posteriori pentru prob de admitere pentru al doilea student
    with logistic_model:
        post_pred_2 = pm.sample_posterior_predictive(trace, samples=5000, var_names=['admission'], model=logistic_model)

    # calculam prob medii si intervalul de 90% HDI pentru al doilea student
    admission_probabilities_2 = np.mean(post_pred_2['admission'], axis=0)
    prob_interval_90_2 = np.percentile(admission_probabilities_2, [5, 95])

    print(f'Intervalul de 90% HDI pentru probabilitatea de admitere pentru al doilea student: {prob_interval_90_2}')
