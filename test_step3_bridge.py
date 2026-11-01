import argparse

import numpy as np
import nnbridge


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--puerto", type=int, default=5050)
    p.add_argument("--node", type=int, required=True)
    p.add_argument("--nodos", type=int, default=1)
    p.add_argument("--rondas", type=int, default=1)
    p.add_argument("--capas", type=int, default=3)
    args = p.parse_args()

    nnbridge.conectar(args.host, args.puerto, args.node, args.nodos, 30)
    try:
        base = float(args.node + 1)

        for ronda in range(args.rondas):
            for capa in range(args.capas):
                enviada = np.full((2, 3), base + capa, dtype=np.float64)

                nnbridge.enviar_matriz(ronda, capa, enviada)
                recibida = nnbridge.recibir_matriz(ronda, capa)

                esperada = float(args.nodos + 1) / 2.0 + capa
                assert recibida.shape == enviada.shape, recibida.shape
                assert np.allclose(recibida, esperada), (recibida, esperada)

                print(f"[nodo {args.node}] ronda {ronda} capa {capa} "f"OK -> {recibida[0, 0]}")
    finally:
        nnbridge.cerrar()

    print(f"[nodo {args.node}] Paso 3 OK")


if __name__ == "__main__":
    main()