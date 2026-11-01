import io
import urllib.request

import numpy as np

URL_PIMA = (
    "https://raw.githubusercontent.com/jbrownlee/Datasets/master/"
    "pima-indians-diabetes.csv"
)

FEATURES_PIMA = [1, 5, 7]  # glucosa, BMI, edad
LABEL_PIMA = 8 # 0 = no diabetico, 1 = diabetico


def _leer_csv(ruta=None, skiprows=0):
    if ruta:
        return np.loadtxt(ruta, delimiter=",", skiprows=skiprows)

    with urllib.request.urlopen(URL_PIMA, timeout=30) as resp:
        texto = resp.read().decode("utf-8")
    return np.loadtxt(io.StringIO(texto), delimiter=",")


def datos_sinteticos(n=600, seed=0):
    rng = np.random.default_rng(seed)
    mitad = n // 2
    x0 = rng.normal([-1.0, -1.0, -1.0], 0.8, size=(mitad, 3))
    x1 = rng.normal([1.0, 1.0, 1.0], 0.8, size=(n - mitad, 3))
    x = np.vstack([x0, x1])
    y = np.concatenate([np.zeros(mitad, dtype=np.int64), np.ones(n - mitad, dtype=np.int64)])
    perm = rng.permutation(len(x))
    return x[perm], y[perm]


def cargar_datos(ruta=None, features=None, label=LABEL_PIMA, skiprows=0):
    if ruta is None:
        try:
            crudo = _leer_csv(None)
        except Exception as exc:
            print(f"[data] no se pudo descargar Pima ({exc}); uso sinteticos")
            return datos_sinteticos()
    else:
        crudo = _leer_csv(ruta, skiprows)

    features = FEATURES_PIMA if features is None else features
    x = crudo[:, features].astype(np.float64)
    y = crudo[:, label].astype(np.int64)
    return x, y


def preparar(nodos, node_id, seed=42, test_frac=0.2, ruta=None, features=None, label=LABEL_PIMA, skiprows=0):
    x, y = cargar_datos(ruta, features, label, skiprows)

    rng = np.random.default_rng(seed)
    train_idx = []
    test_idx = []

    for clase in np.unique(y):
        idx = np.where(y == clase)[0]
        rng.shuffle(idx)
        n_test = max(1, int(round(len(idx) * test_frac)))
        test_idx.extend(idx[:n_test])
        train_idx.extend(idx[n_test:])

    train_idx = np.array(train_idx)
    test_idx = np.array(test_idx)
    rng.shuffle(train_idx)
    rng.shuffle(test_idx)

    mu = x[train_idx].mean(axis=0)
    sigma = x[train_idx].std(axis=0)
    sigma[sigma == 0] = 1.0
    x = (x - mu) / sigma

    partes = np.array_split(train_idx, nodos)
    shard_idx = partes[node_id]

    return x[shard_idx], y[shard_idx], x[test_idx], y[test_idx]