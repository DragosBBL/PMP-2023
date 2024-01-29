import pymc as pm
import arviz as az
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy.stats as stats
import arviz as az
import pytensor as pt

import pandas as pd
df = pd.read_csv('BostonHousing.csv')

with pm.Model() as model:
    intercept = pm.Normal('intercept', mu=0, sigma=10)
    beta_rm = pm.Normal('beta_rm', mu=0, sigma=10)
    beta_crim = pm.Normal('beta_crim', mu=0, sigma=10)
    beta_indus = pm.Normal('beta_indus', mu=0, sigma=10)
    mu = intercept + beta_rm * df['rm'] + beta_crim * df['crim'] + beta_indus * df['indus']
    medv = pm.Normal('medv', mu=mu, sigma=1, observed=df['medv'])

    trace = pm.sample(2000, tune=1000)