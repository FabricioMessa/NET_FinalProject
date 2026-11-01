import numpy as np
import torch

from model import (crear_modelo, extraer_matrices, actualizar_matrices, resumen_modelo)


def test_forma_y_forward():
    modelo = crear_modelo(seed=42)
    x = torch.randn(5, 3)
    y = modelo(x)
    assert y.shape == (5, 2), y.shape
    resumen_modelo(modelo)
    print("forward OK: salida (5, 2)")


def test_misma_semilla():
    a = crear_modelo(seed=123)
    b = crear_modelo(seed=123)
    for pa, pb in zip(a.parameters(), b.parameters()):
        assert torch.equal(pa, pb)
    print("misma semilla -> pesos iniciales identicos")


def test_extraer_actualizar():
    modelo = crear_modelo(seed=7)
    antes = extraer_matrices(modelo)

    assert [m.shape for m in antes] == [(6, 3), (6, 6), (2, 6)]
    assert all(m.dtype == np.float64 for m in antes)

    nuevas = [m + 1.0 for m in antes]
    actualizar_matrices(modelo, nuevas)
    despues = extraer_matrices(modelo)

    for a, d in zip(antes, despues):
        assert np.allclose(d, a + 1.0)
    print("extraer/actualizar OK")


def test_media_estilo_servidor():
    modelos = [crear_modelo(seed=99) for _ in range(3)]

    for i, m in enumerate(modelos):
        actualizar_matrices(m, [w + float(i) for w in extraer_matrices(m)])

    mats = [extraer_matrices(m) for m in modelos]
    media = [np.mean([m[c] for m in mats], axis=0) for c in range(3)]

    actualizar_matrices(modelos[0], media)
    for c in range(3):
        assert np.allclose(extraer_matrices(modelos[0])[c], media[c])
    print("media de 3 modelos OK")


if __name__ == "__main__":
    test_forma_y_forward()
    test_misma_semilla()
    test_extraer_actualizar()
    test_media_estilo_servidor()
    print("Paso 4 OK")