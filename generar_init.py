import numpy as np

from model import crear_modelo, extraer_matrices

pesos = extraer_matrices(crear_modelo(seed=42, dispositivo="cpu"))
np.savez("pesos_iniciales.npz", w0=pesos[0], w1=pesos[1], w2=pesos[2])
print("pesos_iniciales.npz generado")