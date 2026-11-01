import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self, entradas=3, ocultas=6, salidas=2):
        super().__init__()
        self.fc1 = nn.Linear(entradas, ocultas, bias=False)
        self.fc2 = nn.Linear(ocultas, ocultas, bias=False)
        self.fc3 = nn.Linear(ocultas, salidas, bias=False)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


def crear_modelo(seed=42, dispositivo="cpu"):
    torch.manual_seed(seed)
    modelo = Net()
    modelo.to(dispositivo)
    return modelo


def matrices_de_pesos(modelo):
    return [modelo.fc1.weight, modelo.fc2.weight, modelo.fc3.weight]


def extraer_matrices(modelo):
    return [
        p.detach().cpu().numpy().astype(np.float64, copy=True)
        for p in matrices_de_pesos(modelo)
    ]


def actualizar_matrices(modelo, matrices):
    pesos = matrices_de_pesos(modelo)
    if len(matrices) != len(pesos):
        raise ValueError(
            f"se esperaban {len(pesos)} matrices, llegaron {len(matrices)}")

    with torch.no_grad():
        for i, (param, nueva) in enumerate(zip(pesos, matrices)):
            arr = np.ascontiguousarray(nueva, dtype=np.float64)
            if tuple(arr.shape) != tuple(param.shape):
                raise ValueError(
                    f"matriz {i}: shape {arr.shape} != {tuple(param.shape)}")
            param.copy_(
                torch.from_numpy(arr).to(dtype=param.dtype, device=param.device))


def resumen_modelo(modelo):
    for i, p in enumerate(matrices_de_pesos(modelo)):
        print(f"matriz {i}: {tuple(p.shape)} | dtype={p.dtype} | "f"dispositivo={p.device}")