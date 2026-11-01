import argparse
import socket
import struct

import numpy as np

MAGIC = 0x4E4E5731
VERSION = 1
DTYPE_FLOAT64 = 0
HELLO = struct.Struct("<IHHHH")
HEADER = struct.Struct("<IHHIHHIIQ")


def recv_all(sock, n):
    datos = bytearray()
    while len(datos) < n:
        parte = sock.recv(n - len(datos))
        if not parte:
            raise ConnectionError("el servidor cerro la conexion")
        datos.extend(parte)
    return bytes(datos)


def enviar_matriz(sock, node_id, ronda, capa, matriz):
    m = np.ascontiguousarray(np.asarray(matriz, dtype=np.float64))
    filas, columnas = m.shape
    sock.sendall(HEADER.pack(MAGIC, VERSION, node_id, ronda, capa, DTYPE_FLOAT64, filas, columnas, m.nbytes))
    sock.sendall(m.tobytes())


def recibir_matriz(sock):
    magic, version, node_id, ronda, capa, dtype, filas, columnas, nbytes = HEADER.unpack(recv_all(sock, HEADER.size))
    if magic != MAGIC or version != VERSION or dtype != DTYPE_FLOAT64:
        raise ValueError("cabecera invalida")
    datos = recv_all(sock, nbytes)
    return np.frombuffer(datos, dtype=np.float64).reshape(filas, columnas)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--puerto", type=int, default=5000)
    p.add_argument("--node", type=int, required=True)
    p.add_argument("--nodos", type=int, default=3)
    p.add_argument("--rondas", type=int, default=1)
    p.add_argument("--capas", type=int, default=3)
    args = p.parse_args()

    with socket.create_connection((args.host, args.puerto), timeout=30) as sock:
        sock.sendall(HELLO.pack(MAGIC, VERSION, args.node, args.nodos, 0))
        base = float(args.node + 1)

        for ronda in range(args.rondas):
            for capa in range(args.capas):
                enviada = np.full((2, 3), base + capa, dtype=np.float64)
                enviar_matriz(sock, args.node, ronda, capa, enviada)
                recibida = recibir_matriz(sock)

                esperada = float(args.nodos + 1) / 2.0 + capa
                if not np.allclose(recibida, esperada):
                    raise AssertionError(
                        f"capa {capa}: esperaba {esperada}, recibi {recibida}")

                print(f"[nodo {args.node}] ronda {ronda} capa {capa} "f"OK -> {recibida[0, 0]}")

    print(f"[nodo {args.node}] Paso 2 OK")


if __name__ == "__main__":
    main()