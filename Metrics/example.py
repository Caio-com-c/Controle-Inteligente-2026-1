import numpy as np
from metrics import Metrics

t = np.linspace(0, 10, 1000)
y = 1 - np.exp(-t)  # resposta típica
u = np.ones_like(t)
r = np.ones_like(t)

analise = Metrics(t, u, y, r)

resultado = analise.resumo()

for k, v in resultado.items():
    print(f"{k}: {v}")
