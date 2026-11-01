import argparse
import os

import numpy as np
import torch
import torch.nn as nn

import nnbridge
from data import preparar
from model import crear_modelo, extraer_matrices, actualizar_matrices


def obtener_batch(x, y, batch_size, paso):
    n = len(x)
    inicio = (paso * batch_size) % n
    idx = (inicio + np.arange(batch_size)) % n
    return (torch.from_numpy(x[idx]).float(), torch.from_numpy(y[idx]).long())


def evaluar(modelo, x, y):
    modelo.eval()
    with torch.no_grad():
        logits = modelo(torch.from_numpy(x).float())
        pred = torch.argmax(logits, dim=1).numpy()
    return float((pred == y).mean())


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--node", type=int, required=True)
    p.add_argument("--nodos", type=int, default=3)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--puerto", type=int, default=5050)
    p.add_argument("--rondas", type=int, default=200)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--csv", default="data/pima.csv")
    p.add_argument("--features", default=None, help="indices separados por coma, ej: 1,5,7")
    p.add_argument("--label", type=int, default=8)
    p.add_argument("--skiprows", type=int, default=0)
    p.add_argument("--salida", default="resultados")
    p.add_argument("--pasos-locales", type=int, default=5)
    p.add_argument("--init", default=None, help="npz con pesos iniciales compartidos")
    args = p.parse_args()

    features = None
    if args.features:
        features = [int(v) for v in args.features.split(",")]

    x_shard, y_shard, x_test, y_test = preparar(args.nodos, args.node, seed=args.seed, ruta=args.csv, features=features, label=args.label, skiprows=args.skiprows)

    modelo = crear_modelo(seed=args.seed, dispositivo="cpu")

    if args.init:
        datos = np.load(args.init)
        actualizar_matrices(modelo, [datos["w0"], datos["w1"], datos["w2"]])

    criterio = nn.CrossEntropyLoss()
    optimizador = torch.optim.SGD(modelo.parameters(), lr=args.lr)

    print(f"[nodo {args.node}] shard={len(x_shard)} " f"test={len(x_test)} rondas={args.rondas}")

    nnbridge.conectar(args.host, args.puerto, args.node, args.nodos, 60)
    try:
        for ronda in range(args.rondas):
            modelo.train()

            for paso_local in range(args.pasos_locales):
                paso_global = ronda * args.pasos_locales + paso_local
                xb, yb = obtener_batch(x_shard, y_shard, args.batch_size, paso_global)

                optimizador.zero_grad()
                perdida = criterio(modelo(xb), yb)
                perdida.backward()
                optimizador.step()

            promedios = []
            for capa, w in enumerate(extraer_matrices(modelo)):
                nnbridge.enviar_matriz(ronda, capa, w)
                promedios.append(nnbridge.recibir_matriz(ronda, capa))
            actualizar_matrices(modelo, promedios)

            if ronda % 10 == 0 or ronda == args.rondas - 1:
                acc = evaluar(modelo, x_test, y_test)
                print(f"[nodo {args.node}] ronda {ronda + 1}/{args.rondas} " f"loss={perdida.item():.4f} acc={acc:.3f}")
    finally:
        nnbridge.cerrar()

    os.makedirs(args.salida, exist_ok=True)
    pesos = extraer_matrices(modelo)
    ruta_salida = os.path.join(args.salida, f"nodo_{args.node}.npz")
    np.savez(ruta_salida, w0=pesos[0], w1=pesos[1], w2=pesos[2])

    print(f"[nodo {args.node}] accuracy final=" f"{evaluar(modelo, x_test, y_test):.4f} -> {ruta_salida}")

if __name__ == "__main__":
    main()