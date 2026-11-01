import numpy as np
import nnbridge

a = np.arange(6.0, dtype=np.float64).reshape(2, 3)
b = nnbridge.eco_matriz(a)

assert isinstance(b, np.ndarray), type(b)
assert b.shape == (2, 3), b.shape
assert np.array_equal(a, b), (a, b)

print("Paso 1 OK")
print("Entrada:\n", a)
print("Salida:\n", b)
print("Doc del modulo:", nnbridge.__doc__)