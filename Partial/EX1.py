import random
from pgmpy.models import BayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination

# Simulam jocul
P0_wins = 0
P1_wins = 0

# Jucam de 20000 ori
for i in range(20000):

    P0_score = sum([random.choices([0, 1], weights=[2 / 3, 1 / 3])[0] for _ in range(1)])
    P1_score = sum([random.choices([0, 1], weights=[1 / 2, 1 / 2])[0] for _ in range(P0_score + 1)])

    # Determinam castigatorul
    if P0_score >= P1_score:
        P0_wins += 1
    else:
        P1_wins += 1
        
# Printam rezultatele
print(f"P0 wins: {P0_wins}")
print(f"P1 wins: {P1_wins}")

# Construim Bayes Network ( nu mai avem cum sa folosim model)
model = BayesianNetwork([('P0', 'P1')])

# Definim CDP
cpd_p0 = TabularCPD(variable='P0', variable_card=2, values=[[1 / 3], [2 / 3]])

cpd_p1 = TabularCPD(variable='P1', variable_card=2,
                    values=[[1 / 2, 1 / 2],
                            [1 / 2, 1 / 2]],
                    evidence=['P0'],
                    evidence_card=[2])

# Adaugam CDP la Bayesian Network
model.add_cpds(cpd_p0, cpd_p1)

# Verificam daca modelul este valid
model.check_model()

#Determinam probabilitatea lui P0 de a castiga daca scorul lui P1 este 0
infer = VariableElimination(model)
prob_p0 = infer.query(variables=['P0'], evidence={'P1': 0})
print(prob_p0)
