from pgmpy.models import BayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination
import networkx as nx
import matplotlib.pyplot as plt

model = BayesianNetwork([('cutremur', 'incendiu'), ('incendiu', 'alarma'), ('cutremur', 'alarma')])

cpd_cutremur = TabularCPD(variable='cutremur', variable_card=2, values=[[0.9995], [0.0005]])
cpd_incendiu = TabularCPD(variable='incendiu', variable_card=2, values=[[0.99, 0.97], [0.01, 0.03]], evidence=['cutremur'], evidence_card=[2])
cpd_alarma = TabularCPD(variable='alarma', variable_card=2, values=[[0.9999, 0.98, 0.05, 0.02], [0.0001, 0.02, 0.95, 0.98]], evidence=['cutremur', 'incendiu'], evidence_card=[2, 2])

model.add_cpds(cpd_cutremur, cpd_incendiu, cpd_alarma)
assert model.check_model()

infer = VariableElimination(model)
result_cutremur_given_alarma = infer.query(variables=['cutremur'], evidence={'alarma': 1})
print(result_cutremur_given_alarma)

result_incendiu_without_alarma = infer.query(variables=['incendiu'])
print(result_incendiu_without_alarma)

pos = nx.circular_layout(model)
nx.draw(model, pos=pos, with_labels=True, node_size=4000, font_weight='bold', node_color='skyblue')
plt.show()
