# Guía completa del código — Trabajo Final de Redes

**Curso:** Networks and Communications  
**Proyecto:** Entrenamiento distribuido de una red neuronal con Python/PyTorch + C++ + sockets TCP  
**Uso de esta guía:** preparación para la defensa y para la modificación individual del 40%.

> Esta guía explica qué hace cada archivo, cada bloque y cada línea importante del proyecto.
> Los números de línea corresponden a la versión actual del código. Si editas un archivo y las
> líneas se mueven, busca por el nombre de la función o por el fragmento de código indicado.

---

## Índice

1. [Resumen ejecutivo](#1-resumen-ejecutivo)
2. [Mapa del proyecto](#2-mapa-del-proyecto)
3. [Conceptos previos](#3-conceptos-previos)
4. [Flujo completo de una ronda](#4-flujo-completo-de-una-ronda)
5. [`cpp/socket_utils.hpp`](#5-cppsocket_utilshpp)
6. [`cpp/protocol.hpp`](#6-cppprotocolhpp)
7. [`cpp/servidor_central.cpp`](#7-cppservidor_centralcpp)
8. [`cpp/bridge.cpp`](#8-cppbridgecpp)
9. [`model.py`](#9-modelpy)
10. [`data.py`](#10-datapy)
11. [`entrenar_nodo.py`](#11-entrenar_nodopy)
12. [`generar_init.py`](#12-generar_initpy)
13. [Pruebas incrementales](#13-pruebas-incrementales)
14. [Compilación y build](#14-compilación-y-build)
15. [Datos y artefactos](#15-datos-y-artefactos)
16. [Ejecución localhost y LAN](#16-ejecución-localhost-y-lan)
17. [Protocolo y secuencia](#17-protocolo-y-secuencia)
18. [Errores y depuración](#18-errores-y-depuración)
19. [Guía del 40%: modificaciones probables](#19-guía-del-40-modificaciones-probables)
20. [Glosario](#20-glosario)
21. [Preguntas de defensa](#21-preguntas-de-defensa)
22. [Checklist final](#22-checklist-final)

---

## 1. Resumen ejecutivo

El proyecto entrena una red neuronal pequeña de forma **distribuida** entre 3 nodos:

- Cada nodo es un proceso Python que usa **PyTorch**.
- La red es `3 → 6 → 6 → 2`, con ReLU en las capas ocultas y sin bias.
- Cada nodo entrena con una **partición (shard)** distinta del dataset.
- Después de varios pasos locales, cada nodo extrae sus **3 matrices de pesos**:
  - `fc1.weight`: `(6, 3)`
  - `fc2.weight`: `(6, 6)`
  - `fc3.weight`: `(2, 6)`
- Las matrices viajan por **TCP** desde C++ (`nnbridge`) hasta un **servidor central** en C++.
- El servidor espera las 3 matrices de cada capa, calcula la **media celda por celda** y las devuelve.
- Cada nodo actualiza sus pesos con la media y continúa con la siguiente ronda.
- Al final, los 3 nodos tienen exactamente los mismos pesos y la misma accuracy.

Resultado de la corrida local validada:

- 3 nodos, 200 rondas, 5 pasos locales, `lr=0.2`, `batch=32`.
- Accuracy: **0.7532**.
- Matriz de confusión: `[[86, 14], [24, 30]]`.
- Pesos finales idénticos en los 3 nodos.

La parte de **C/C++** es la comunicación y la media. La parte de **Python** es la red, el
entrenamiento y la extracción/actualización de matrices. La unión se hace con **pybind11**.

---

## 2. Mapa del proyecto

```text
FinalProject/
├── CMakeLists.txt                  # Compila nnbridge (pybind11) y servidor_central
├── requirements.txt                # pybind11, numpy y torch con versiones
├── .gitignore
├── data/
│   └── pima.csv                    # Dataset de prueba (768 filas × 9 columnas)
├── pesos_iniciales.npz             # Pesos iniciales compartidos (w0, w1, w2)
├── resultados/
│   ├── nodo_0.npz
│   ├── nodo_1.npz
│   └── nodo_2.npz                 # Pesos finales de cada nodo
├── cpp/
│   ├── socket_utils.hpp            # Sockets portables POSIX/Winsock
│   ├── protocol.hpp                # Cabeceras binarias y envío/recepción
│   ├── bridge.cpp                  # Módulo Python "nnbridge" hecho en C++
│   └── servidor_central.cpp        # Servidor central: media de matrices
├── scripts/
│   └── build.sh                    # Configura y compila con CMake
├── model.py                        # Red 3→6→6→2 y funciones de matrices
├── data.py                         # Carga, normalización, split y shards
├── entrenar_nodo.py                # Loop de entrenamiento distribuido
├── generar_init.py                 # Genera pesos_iniciales.npz
├── test_step1.py                   # Prueba pybind11 / eco_matriz
├── test_step2_cliente.py           # Prueba del protocolo con sockets "crudos"
├── test_step3_bridge.py            # Prueba del bridge contra el servidor
└── test_step4_model.py             # Prueba de extracción/actualización y media
```

### Quién llama a quién

```text
entrenar_nodo.py
  ├── data.preparar()              → carga CSV, normaliza, divide y elige shard
  ├── model.crear_modelo()         → crea la red PyTorch
  ├── model.extraer_matrices()     → pesos PyTorch → numpy float64
  ├── nnbridge.conectar()          → C++ abre socket TCP y envía Hello
  ├── nnbridge.enviar_matriz()     → C++ serializa y envía una matriz
  ├── nnbridge.recibir_matriz()    → C++ recibe y devuelve numpy
  └── model.actualizar_matrices()  → media numpy → pesos PyTorch

servidor_central
  ├── recibe Hello de cada nodo
  ├── recibe matrices por (ronda, capa)
  ├── calcula media celda por celda
  └── devuelve la media a cada nodo
```

---

## 3. Conceptos previos

### 3.1 Entrenamiento distribuido y Federated Averaging

En entrenamiento centralizado, un solo proceso ve todos los datos y actualiza un solo modelo.
En este proyecto hay 3 procesos, cada uno con su propio modelo y su propio shard. Para que los
modelos no se alejen entre sí, después de un grupo de pasos locales se **promedian los pesos**:

```text
W_global = (W_nodo0 + W_nodo1 + W_nodo2) / 3
```

Ese promedio se hace **celda por celda**, no con multiplicación de matrices. Si en una celda los
nodos tienen `3`, `2` y `10`, el resultado es `(3 + 2 + 10) / 3 = 5`.

Este esquema se conoce como **Federated Averaging (FedAvg)**: entrenamiento local + promedio de
parámetros. En este proyecto se promedian **pesos**, no gradientes.

### 3.2 Matrices de pesos y shapes

PyTorch guarda el peso de una capa `nn.Linear(in_features, out_features)` con shape
`(out_features, in_features)`:

| Capa | Operación | Shape de `weight` | Parámetros |
|---|---|---|---|
| `fc1` | 3 → 6 | `(6, 3)` | 18 |
| `fc2` | 6 → 6 | `(6, 6)` | 36 |
| `fc3` | 6 → 2 | `(2, 6)` | 12 |
| **Total** | | | **66** |

El dibujo del profesor muestra `[3,6]`, `[6,6]` y `[6,2]` porque usa la convención matemática
`entrada × salida`. PyTorch usa la transpuesta. No es un error: mientras los 3 nodos usen el
mismo orden y los mismos shapes, la media funciona.

### 3.3 Shard, batch, paso local y ronda

- **Shard:** parte del dataset que le toca a un nodo. Con 614 muestras de train y 3 nodos,
  los shards son de 205, 205 y 204 muestras.
- **Batch:** grupo de muestras que se pasa en un `forward/backward`. Aquí `batch_size=32`.
- **Paso local:** una actualización de pesos con un batch.
- **Ronda:** grupo de `pasos_locales` (5 por defecto) y luego **una sincronización** con el
  servidor. En cada ronda se envían las 3 matrices y se recibe la media de las 3.

Con 200 rondas y 5 pasos locales, cada nodo hace 1000 actualizaciones locales.

### 3.4 pybind11

pybind11 es una librería de C++ que permite crear un **módulo importable desde Python**. En este
proyecto, el archivo `cpp/bridge.cpp` define:

```cpp
PYBIND11_MODULE(nnbridge, m) { ... }
```

Eso produce un archivo `nnbridge.cpython-314-darwin.so` (macOS), `nnbridge...so` (Linux) o
`nnbridge...pyd` (Windows). Desde Python se usa así:

```python
import nnbridge
nnbridge.conectar("127.0.0.1", 5050, 0, 3, 60)
nnbridge.enviar_matriz(0, 0, matriz_numpy)
media = nnbridge.recibir_matriz(0, 0)
nnbridge.cerrar()
```

pybind11 convierte automáticamente entre `numpy.ndarray` y `py::array_t<double>`.

### 3.5 TCP y framing

TCP entrega un **flujo de bytes**, no mensajes. Si envías dos matrices seguidas, el receptor no
sabe dónde termina una y empieza la otra. Por eso el protocolo usa:

1. Una **cabecera de tamaño fijo** (`MsgHeader`, 32 bytes) con `filas`, `columnas` y `nbytes`.
2. El **payload** con los datos de la matriz.

Además, `send()` y `recv()` pueden transferir menos bytes de los pedidos. Por eso existen
`send_all()` y `recv_all()`, que repiten la operación hasta completar el bloque.

### 3.6 float32 vs float64

- PyTorch usa `float32` por defecto en los pesos.
- El protocolo y `py::array_t<double>` usan `float64`.
- `model.extraer_matrices()` convierte a `float64` antes de enviar.
- `model.actualizar_matrices()` convierte de vuelta a `float32` al copiar al parámetro.

Esto evita perder precisión en la media y mantiene simple el lado C++.

### 3.7 bias = False

La red usa `bias=False` en las 3 capas. Así el modelo tiene **exactamente 3 matrices** y todo el
modelo queda sincronizado al promediar esas 3 matrices. Si se usaran bias, habría 6 parámetros
(3 pesos + 3 bias) y los bias quedarían distintos en cada nodo, porque el servidor solo promedia
las 3 matrices de pesos.

---

## 4. Flujo completo de una ronda

```mermaid
sequenceDiagram
    participant P0 as Python nodo 0
    participant B0 as nnbridge (C++)
    participant S as ServidorCentral (C++)
    participant B1 as nnbridge nodo 1
    participant B2 as nnbridge nodo 2

    Note over P0,B2: Handshake inicial (una vez)
    P0->>B0: conectar(host, puerto, node_id=0, num_nodos=3)
    B0->>S: Hello(magic, version, node_id=0, num_nodos=3)
    B1->>S: Hello(... node_id=1 ...)
    B2->>S: Hello(... node_id=2 ...)
    S-->>B0: nodo 0 aceptado
    S-->>B1: nodo 1 aceptado
    S-->>B2: nodo 2 aceptado

    Note over P0,B2: Cada ronda
    loop 5 pasos locales
        P0->>P0: forward + backward + optimizer.step()
    end
    P0->>B0: extraer_matrices() → [W0, W1, W2]
    B0->>S: MsgHeader + W0 (capa 0)
    B1->>S: MsgHeader + W0 (capa 0)
    B2->>S: MsgHeader + W0 (capa 0)
    S->>S: media celda por celda de W0
    S-->>B0: W0_media
    S-->>B1: W0_media
    S-->>B2: W0_media
    B0->>P0: recibir_matriz() → W0_media

    Note over P0,B2: Se repite para capa 1 y capa 2
    P0->>P0: actualizar_matrices([W0_media, W1_media, W2_media])
end
```

### Orden exacto que deben seguir los 3 nodos

Por cada ronda `r`:

```text
para capa = 0, 1, 2:
    enviar_matriz(r, capa, W[capa])
    W_nueva[capa] = recibir_matriz(r, capa)
actualizar_matrices(modelo, W_nueva)
```

El servidor hace exactamente el orden inverso:

```text
para capa = 0, 1, 2:
    para nodo = 0, 1, 2:
        recibir matriz (r, capa) del nodo
    media = promedio de las 3 matrices
    para nodo = 0, 1, 2:
        enviar media al nodo
```

Si un nodo se salta una capa o cambia el orden, el servidor quedará esperando y se producirá un
timeout o un error de ronda/capa.

---

## 5. `cpp/socket_utils.hpp`

**Responsabilidad:** abstraer las diferencias entre macOS/Linux (POSIX) y Windows (Winsock), y
ofrecer utilidades comunes: tipo de socket, cierre, mensajes de error y timeouts.

| Líneas | Qué hace |
|---|---|
| 1 | `#pragma once`: evita incluir el archivo dos veces. |
| 3-6 | Includes estándar: `cerrno`, `cstring`, `stdexcept`, `string`. |
| 8-11 | En Windows incluye `winsock2.h`, `ws2tcpip.h` y enlaza `ws2_32.lib`. |
| 12-22 | En POSIX incluye sockets, `netdb`, `netinet`, `signal`, `unistd`, `sys/time`. |
| 24 | Abre `namespace red` para evitar colisiones con otros nombres. |
| 26-32 | Define `socket_t`: `SOCKET` en Windows, `int` en POSIX. Define `SOCKET_INVALIDO`. |
| 34-43 | `InicializarWinsock`: RAII que llama `WSAStartup` al crear y `WSACleanup` al destruir. Solo Windows. |
| 45-52 | `inicializar_sockets()`: en Windows crea el singleton de Winsock; en POSIX ignora `SIGPIPE` para que escribir a un socket cerrado no mate el proceso. |
| 54-61 | `cerrar_socket(s)`: cierra con `closesocket` o `close`, e ignora el socket inválido. |
| 63-69 | `error_socket()`: devuelve el último error del sistema como `std::string`. |
| 71-83 | `configurar_timeout(s, segundos)`: configura `SO_RCVTIMEO` y `SO_SNDTIMEO` para que `recv`/`send` no bloqueen para siempre. En Windows usa `DWORD` en milisegundos; en POSIX usa `timeval`. |
| 85 | Cierra `namespace red`. |

### Por qué existe este archivo

Sin esta abstracción, el código de sockets tendría `#ifdef _WIN32` repartido por todas partes.
Aquí se concentra una sola vez. `protocol.hpp`, `bridge.cpp` y `servidor_central.cpp` usan
`red::socket_t`, `red::cerrar_socket`, etc., y funcionan en los 3 sistemas.

### Detalle importante: `SIGPIPE`

En Linux/macOS, si un proceso escribe en un socket que el otro extremo cerró, el sistema puede
enviar la señal `SIGPIPE` y matar el proceso. `signal(SIGPIPE, SIG_IGN)` evita eso; en su lugar,
`send()` devuelve error y el código lanza una excepción controlada.

---

## 6. `cpp/protocol.hpp`

**Responsabilidad:** definir el formato binario de los mensajes y las funciones para enviar y
recibir enteros, cabeceras y matrices.

| Líneas | Qué hace |
|---|---|
| 1 | `#pragma once`. |
| 3 | Incluye `socket_utils.hpp` para tener `socket_t` y utilidades. |
| 5-8 | Includes: `cstdint`, `cstring`, `stdexcept`, `vector`. |
| 10 | `namespace red`. |
| 12 | `MAGIC = 0x4E4E5731`: número mágico que identifica el protocolo ("NNW1"). |
| 13 | `VERSION = 1`: versión del protocolo. |
| 14 | `DTYPE_FLOAT64 = 0`: código para matrices `double`. |
| 16 | `#pragma pack(push, 1)`: empaqueta las estructuras sin relleno. |
| 17-23 | `Hello`: mensaje de handshake (12 bytes). |
| 25-35 | `MsgHeader`: cabecera de una matriz (32 bytes). |
| 36 | `#pragma pack(pop)`: restaura el empaquetado normal. |
| 38-39 | `static_assert` que verifica los tamaños en tiempo de compilación. |
| 41-53 | `send_all`: envía exactamente `n` bytes, repitiendo `send` si hace falta. |
| 55-68 | `recv_all`: recibe exactamente `n` bytes, repitiendo `recv` si hace falta. |
| 70-73 | `enviar_hello`: construye y envía un `Hello`. |
| 75-81 | `recibir_hello`: recibe un `Hello` y valida `magic` y `version`. |
| 83-87 | `enviar_matriz`: construye la cabecera con shape y `nbytes`, envía cabecera y payload. |
| 89-100 | `recibir_matriz`: recibe cabecera, valida, redimensiona el vector y recibe el payload. |
| 102 | Cierra `namespace red`. |

### 6.1 Layout de `Hello` (12 bytes)

| Campo | Tipo | Bytes | Significado |
|---|---|---|---|
| `magic` | `uint32_t` | 4 | Debe ser `0x4E4E5731`. |
| `version` | `uint16_t` | 2 | Debe ser `1`. |
| `node_id` | `uint16_t` | 2 | Índice del nodo: 0, 1 o 2. |
| `num_nodos` | `uint16_t` | 2 | Total de nodos: 3. |
| `reservado` | `uint16_t` | 2 | Reservado para futuras extensiones. |

### 6.2 Layout de `MsgHeader` (32 bytes)

| Campo | Tipo | Bytes | Significado |
|---|---|---|---|
| `magic` | `uint32_t` | 4 | Identificador del protocolo. |
| `version` | `uint16_t` | 2 | Versión. |
| `node_id` | `uint16_t` | 2 | Nodo que envía. |
| `ronda` | `uint32_t` | 4 | Número de ronda. |
| `capa` | `uint16_t` | 2 | 0 = `fc1`, 1 = `fc2`, 2 = `fc3`. |
| `dtype` | `uint16_t` | 2 | 0 = `float64`. |
| `filas` | `uint32_t` | 4 | Filas de la matriz. |
| `columnas` | `uint32_t` | 4 | Columnas de la matriz. |
| `nbytes` | `uint64_t` | 8 | `filas * columnas * 8`. |

### 6.3 Ejemplo de un mensaje

Para `fc1.weight` de shape `(6, 3)`:

- `filas = 6`, `columnas = 3`.
- `nbytes = 6 * 3 * 8 = 144`.
- Mensaje total = 32 bytes de cabecera + 144 bytes de datos = **176 bytes**.

### 6.4 `send_all` y `recv_all`

```cpp
while (enviado < n) {
    r = send(s, p + enviado, n - enviado, 0);
    if (r <= 0) throw ...;
    enviado += r;
}
```

`send` puede enviar menos de lo pedido. El bucle avanza el puntero y acumula hasta completar.
`recv_all` hace lo mismo, pero distingue:

- `r == 0`: el otro extremo cerró la conexión.
- `r < 0`: error (timeout, conexión reseteada, etc.).

### 6.5 Por qué se valida `nbytes`

Un `nbytes` corrupto o mal calculado podría hacer que el receptor reserve memoria incorrecta o
lea datos de otra matriz. La comprobación:

```cpp
const uint64_t esperado = filas * columnas * sizeof(double);
if (h.nbytes != esperado) throw ...;
```

garantiza que la cabecera y el payload sean coherentes.

---

## 7. `cpp/servidor_central.cpp`

**Responsabilidad:** escuchar conexiones, aceptar exactamente `num_nodos` clientes, recibir sus
matrices por ronda y capa, calcular la media y devolverla.

| Líneas | Qué hace |
|---|---|
| 1-8 | Includes: protocolo, sockets, enteros, `iostream`, `stdexcept`, `vector`. |
| 10 | `namespace { ... }` anónimo: funciones internas del archivo. |
| 11-33 | `crear_listen_socket(puerto)`: crea socket TCP, `SO_REUSEADDR`, `bind` a `INADDR_ANY`, `listen` con backlog 16. |
| 14 | `::socket(AF_INET, SOCK_STREAM, 0)`: socket IPv4/TCP. |
| 19 | `SO_REUSEADDR`: permite reutilizar el puerto al reiniciar. |
| 21-24 | `sockaddr_in`: familia IPv4, cualquier interfaz, puerto en orden de red con `htons`. |
| 26-27 | `bind`: asocia el socket al puerto. |
| 29-30 | `listen`: pone el socket en modo escucha. |
| 35-44 | `aceptar_con_timeout`: configura timeout y llama `accept`. |
| 48-53 | `main`: lee `puerto`, `num_nodos`, `rondas`, `num_capas` de `argv`, con defaults. |
| 55-56 | Valida que los parámetros sean positivos. |
| 58-59 | Crea el socket de escucha e imprime la configuración. |
| 61-83 | Bucle de aceptación de nodos. |
| 61 | `nodos` es un vector de sockets indexado por `node_id`; inicia en inválido. |
| 63 | Repite hasta tener `num_nodos` conectados. |
| 64 | Acepta una conexión con timeout de 30 s. |
| 65-66 | Si falla, error de timeout. |
| 68 | Configura timeout de 30 s para el handshake del cliente aceptado. |
| 69 | Recibe el `Hello`. |
| 70-74 | Valida `num_nodos` y rango de `node_id`; si es inválido, cierra y sigue. |
| 75-79 | Si el `node_id` ya está ocupado, rechaza el duplicado. |
| 80-82 | Guarda el socket en `nodos[node_id]`, incrementa el contador y loguea. |
| 85 | Configura timeout de 60 s para la fase de entrenamiento. |
| 87-117 | Doble bucle de rondas y capas. |
| 89 | `matrices[num_nodos]`: una matriz por nodo. |
| 90 | `filas`, `columnas`: shape que se valida entre nodos. |
| 92-105 | Recibe de cada nodo en orden 0, 1, 2. |
| 93 | `recibir_matriz` llena `matrices[i]` y devuelve la cabecera. |
| 94-95 | Valida que el `node_id` de la cabecera coincida con el índice. |
| 96-98 | Valida que la ronda y la capa sean las esperadas. |
| 99-104 | El primer nodo define el shape; los demás deben coincidir. |
| 107-111 | Calcula la media celda por celda. |
| 108-110 | Suma las matrices de todos los nodos. |
| 111 | Divide cada celda entre `num_nodos`. |
| 113-114 | Envía la media a cada nodo, en orden 0, 1, 2. |
| 116 | Loguea la ronda completada. |
| 119-122 | Cierra sockets, cierra el socket de escucha, imprime "listo" y retorna 0. |
| 123-126 | Captura cualquier excepción, imprime el error y retorna 1. |

### 7.1 Algoritmo de la media

Para cada capa y cada celda `k`:

```text
media[k] = (M0[k] + M1[k] + M2[k]) / 3
```

No se usa `np.mean`; se implementa en C++ con dos bucles:

```cpp
for (int i = 0; i < num_nodos; ++i)
    for (size_t k = 0; k < media.size(); ++k)
        media[k] += matrices[i][k];

for (double& v : media) v /= num_nodos;
```

Esto es equivalente a `np.mean([M0, M1, M2], axis=0)`.

### 7.2 Por qué el servidor usa orden fijo y no hilos

El servidor es **single-thread**: lee primero del nodo 0, luego del 1 y luego del 2, siempre en el
mismo orden. Los clientes también envían en ese orden. Esto evita condiciones de carrera y
deadlocks, y es suficiente para 3 nodos. La desventaja es que si un nodo se cae, el servidor
queda esperando hasta el timeout.

### 7.3 Manejo de duplicados

Si llega un segundo cliente con el mismo `node_id`, el servidor lo rechaza:

```cpp
if (nodos[h.node_id] != red::SOCKET_INVALIDO) {
    std::cerr << "[servidor] node_id duplicado " << h.node_id << std::endl;
    red::cerrar_socket(s);
    continue;
}
```

Por eso, si lanzas dos juegos de 3 clientes, los duplicados fallan. Debe haber exactamente un
cliente por `node_id`.

### 7.4 Timeouts

- `aceptar_con_timeout(listen_sock, 30)`: 30 s para aceptar.
- Handshake: 30 s.
- Entrenamiento: 60 s por operación de socket.

Si un nodo no envía a tiempo, `recv_all` falla y el servidor termina con error. Para reiniciar,
hay que cerrar todo y volver a arrancar limpio.

---

## 8. `cpp/bridge.cpp`

**Responsabilidad:** crear el módulo Python `nnbridge` y exponer funciones C++ que abren un
socket, envían matrices y reciben matrices.

| Líneas | Qué hace |
|---|---|
| 1-2 | Includes de pybind11: módulo y soporte NumPy. |
| 4-5 | Incluye `protocol.hpp` y `socket_utils.hpp`. |
| 7-11 | Includes estándar. |
| 13 | Alias `namespace py = pybind11`. |
| 15-51 | `namespace` anónimo con estado global y `conectar_tcp`. |
| 17 | `g_socket`: socket global del proceso Python. |
| 18 | `g_node_id`: id del nodo actual. |
| 20-49 | `conectar_tcp(host, puerto)`: usa `getaddrinfo`, prueba direcciones y conecta. |
| 21 | Inicializa sockets (Winsock o SIGPIPE). |
| 23-25 | `addrinfo`: IPv4, TCP. |
| 29 | `getaddrinfo`: resuelve host y puerto. |
| 34-42 | Recorre resultados, crea socket y llama `connect`. |
| 43 | `freeaddrinfo`: libera la lista. |
| 45-46 | Si ninguna dirección funcionó, lanza error. |
| 53-69 | `eco_matriz`: función de prueba que copia una matriz 2-D. |
| 56 | `request()` obtiene `py::buffer_info` (shape, dtype, puntero). |
| 57-58 | Valida que sea 2-D. |
| 63-64 | Crea la matriz de salida y su buffer. |
| 66 | Copia los datos con `memcpy`. |
| 68 | Devuelve la nueva matriz. |
| 71-89 | `py_conectar`: valida estado, conecta, configura timeout y envía `Hello`. |
| 72-73 | Si ya hay conexión, error. |
| 74-75 | Valida `node_id` y `num_nodos`. |
| 77 | Abre el socket. |
| 78 | Configura timeout. |
| 80-86 | Envía `Hello`; si falla, cierra y propaga el error. |
| 88 | Guarda `g_node_id`. |
| 91-97 | `py_cerrar`: cierra el socket y resetea el estado. |
| 99-113 | `py_enviar_matriz`: valida conexión, ronda, capa y shape; envía. |
| 108 | Obtiene buffer de la matriz NumPy. |
| 109-110 | Exige 2-D. |
| 112 | Llama a `red::enviar_matriz` con shape y puntero. |
| 115-136 | `py_recibir_matriz`: recibe cabecera, valida node/ronda/capa, crea NumPy y copia. |
| 120 | `red::recibir_matriz` llena `datos`. |
| 122-126 | Valida que la respuesta corresponda al nodo y a la ronda/capa pedidas. |
| 128-131 | Crea `py::array_t<double>` con el shape recibido. |
| 134 | Copia el payload al array de NumPy. |
| 138-144 | `py_sincronizar_matriz`: envía y recibe en una sola llamada. |
| 146-160 | `PYBIND11_MODULE(nnbridge, m)`: registra las funciones con sus nombres Python y argumentos. |
| 151 | `conectar(host, puerto, node_id, num_nodos, timeout_seg=30)`. |
| 153 | `enviar_matriz(ronda, capa, matriz)`. |
| 155 | `recibir_matriz(ronda, capa)`. |
| 157 | `sincronizar_matriz(ronda, capa, matriz)`. |
| 159 | `cerrar()`. |

### 8.1 Estado global

`g_socket` y `g_node_id` son globales del módulo. Esto significa que **cada proceso Python tiene
una sola conexión activa**. No se pueden abrir dos conexiones simultáneas desde el mismo proceso
con este diseño. Para este proyecto es suficiente: cada nodo es un proceso independiente.

### 8.2 `py::array_t<double, c_style | forcecast>`

- `double`: tipo de dato esperado (`float64`).
- `c_style`: exige memoria contigua en orden C.
- `forcecast`: si llega `float32` u otro tipo, pybind11 lo convierte a `double`.

Así el C++ siempre recibe una matriz contigua y del tipo correcto, sin tener que copiar a mano en
Python.

### 8.3 Por qué se valida `h.node_id`, `h.ronda` y `h.capa`

El servidor devuelve las medias en el mismo orden en que las recibe. Si el cliente pidiera una
capa distinta, o si hubiera un desfase, la validación lo detecta antes de actualizar el modelo
con datos equivocados.

### 8.4 GIL y bloqueo

Las funciones de `nnbridge` son llamadas desde Python y mantienen el **GIL** (Global Interpreter
Lock) mientras el C++ espera en `recv`. En este proyecto no hay hilos Python concurrentes, así que
no es un problema. Si se quisiera paralelizar, habría que liberar el GIL con
`py::gil_scoped_release` alrededor de las llamadas bloqueantes.

---

## 9. `model.py`

**Responsabilidad:** definir la red, crearla con una semilla, extraer las 3 matrices y
actualizarlas.

| Líneas | Qué hace |
|---|---|
| 1-4 | Imports: NumPy, PyTorch, `nn`, `F`. |
| 7 | `class Net(nn.Module)`: red neuronal. |
| 8 | `__init__(entradas=3, ocultas=6, salidas=2)`: dimensiones. |
| 9 | `super().__init__()`: inicializa `nn.Module`. |
| 10 | `fc1 = nn.Linear(3, 6, bias=False)`. |
| 11 | `fc2 = nn.Linear(6, 6, bias=False)`. |
| 12 | `fc3 = nn.Linear(6, 2, bias=False)`. |
| 14-17 | `forward(x)`: `ReLU(fc1) → ReLU(fc2) → fc3`. La salida son 2 logits. |
| 20-24 | `crear_modelo(seed=42, dispositivo="cpu")`: fija semilla, crea `Net`, lo mueve al dispositivo. |
| 21 | `torch.manual_seed(seed)`: reproducibilidad. |
| 22 | `modelo = Net()`. |
| 23 | `modelo.to(dispositivo)`. |
| 27-28 | `matrices_de_pesos(modelo)`: devuelve las 3 matrices en orden fijo. |
| 31-35 | `extraer_matrices(modelo)`: convierte cada peso a NumPy `float64` copiado. |
| 33 | `p.detach().cpu().numpy().astype(np.float64, copy=True)`. |
| 38-51 | `actualizar_matrices(modelo, matrices)`: valida y copia las matrices al modelo. |
| 40-42 | Valida que lleguen exactamente 3 matrices. |
| 44 | `with torch.no_grad()`: no construir grafo de autograd. |
| 45 | Recorre pesos y matrices nuevas. |
| 46 | `np.ascontiguousarray(..., dtype=np.float64)`: memoria contigua y tipo correcto. |
| 47-49 | Valida que el shape coincida con el parámetro. |
| 50-51 | `param.copy_(torch.from_numpy(arr).to(...))`: copia in-place, casteando a `float32` y al dispositivo. |
| 54-56 | `resumen_modelo`: imprime shape, dtype y dispositivo de cada matriz. |

### 9.1 Forward y shapes

Con un batch de `B` muestras:

```text
x:        (B, 3)
fc1(x):   (B, 6)
fc2(.):   (B, 6)
fc3(.):   (B, 2)   ← logits
```

`CrossEntropyLoss` espera logits `(B, 2)` y etiquetas `(B,)` con valores `0` o `1`.

### 9.2 `detach().cpu().numpy()`

- `detach()`: separa el tensor del grafo de autograd; no se necesita gradiente para enviar.
- `cpu()`: mueve a CPU; NumPy no puede leer tensores en GPU/MPS directamente.
- `numpy()`: convierte a `ndarray`.
- `astype(np.float64, copy=True)`: copia para no compartir memoria con el tensor y usar el tipo
  del protocolo.

### 9.3 `param.copy_` vs `param.data = ...`

`param.copy_` copia **in-place**, manteniendo la identidad del parámetro que el optimizador ya
tiene registrado. Hacer `param.data = nuevo_tensor` también funciona, pero `copy_` es más directo
y no rompe referencias. Se usa dentro de `torch.no_grad()` para no registrar la operación.

### 9.4 Por qué el orden es fijo

`matrices_de_pesos` siempre devuelve `[fc1.weight, fc2.weight, fc3.weight]`. Ese orden es el que
el servidor espera como capas `0`, `1` y `2`. Si se cambia el orden, la media mezclaría capas de
distintos tamaños y fallaría la validación de shapes.

---

## 10. `data.py`

**Responsabilidad:** cargar el dataset, normalizarlo, dividirlo en train/test de forma
estratificada y repartir el train en shards.

| Líneas | Qué hace |
|---|---|
| 1-4 | Imports: `io`, `urllib`, NumPy. |
| 6-9 | URL de Pima Indians Diabetes. |
| 11 | `FEATURES_PIMA = [1, 5, 7]`: glucosa, BMI, edad. |
| 12 | `LABEL_PIMA = 8`: columna de etiqueta 0/1. |
| 15-21 | `_leer_csv`: lee un CSV local o descarga Pima y lo parsea con `np.loadtxt`. |
| 17 | Si hay ruta, `np.loadtxt(ruta, delimiter=",", skiprows=skiprows)`. |
| 19-21 | Si no hay ruta, descarga y parsea desde memoria. |
| 24-32 | `datos_sinteticos`: genera 2 clases gaussianas de 3 features. |
| 35-48 | `cargar_datos`: obtiene la matriz cruda y selecciona features y etiqueta. |
| 36-41 | Si no hay ruta, intenta descargar; si falla, usa datos sintéticos. |
| 42-43 | Si hay ruta, lee el archivo. |
| 45 | Si `features is None`, usa las columnas de Pima. |
| 46 | `x = crudo[:, features].astype(np.float64)`. |
| 47 | `y = crudo[:, label].astype(np.int64)`. |
| 51-78 | `preparar`: split estratificado, normalización y shard. |
| 52 | Carga datos. |
| 54 | `rng = np.random.default_rng(seed)`. |
| 55-56 | Listas de índices de train y test. |
| 58-63 | Por cada clase, mezcla sus índices y aparta `test_frac` para test. |
| 61 | `n_test = max(1, int(round(len(idx) * test_frac)))`. |
| 62-63 | Agrega índices a test/train. |
| 65-68 | Convierte a arrays y mezcla train/test. |
| 70-73 | Calcula media y desviación **solo con train** y normaliza todo `x`. |
| 72 | Si `sigma == 0`, usa 1 para evitar división por cero. |
| 75 | `np.array_split(train_idx, nodos)`: divide los índices de train en N partes. |
| 76 | `shard_idx = partes[node_id]`: el shard de este nodo. |
| 78 | Devuelve `x_shard, y_shard, x_test, y_test`. |

### 10.1 Dataset Pima

El CSV tiene 768 filas y 9 columnas:

| Columna | Nombre aproximado |
|---|---|
| 0 | Embarazos |
| 1 | Glucosa |
| 2 | Presión arterial |
| 3 | Grosor de piel |
| 4 | Insulina |
| 5 | BMI |
| 6 | Función de pedigrí de diabetes |
| 7 | Edad |
| 8 | Resultado (0/1) |

El proyecto usa solo `[1, 5, 7]` como entradas y `8` como etiqueta, para cumplir el modelo de
3 entradas y 2 salidas.

### 10.2 Split estratificado

El split no es un simple 80/20 aleatorio: se hace **por clase**. Para cada clase:

1. Se mezclan sus índices.
2. Se aparta el 20% para test.
3. El resto va a train.

Así se mantiene la proporción de positivos/negativos en train y test. En la corrida validada:

- Train: 614 muestras.
- Test: 154 muestras (54 positivas, 100 negativas).
- Shards: 205, 205 y 204.

### 10.3 Normalización

Se calcula `mu` y `sigma` con el train y se aplica a todo el dataset:

```text
x_normalizado = (x - mu) / sigma
```

Es importante que los 3 nodos usen el mismo CSV, el mismo `seed`, las mismas features y el mismo
`label`. Si no, cada nodo calcularía un split distinto y el servidor recibiría shapes o
distribuciones incompatibles.

### 10.4 Cómo cambiar de dataset

El código acepta un CSV arbitrario:

```bash
python entrenar_nodo.py \
  --csv data/otro.csv \
  --features 0,1,2 \
  --label 3 \
  --skiprows 1
```

- `--features`: índices de columnas separados por coma.
- `--label`: índice de la columna de etiqueta.
- `--skiprows`: 1 si el CSV tiene cabecera.

Si el nuevo dataset no tiene 3 features o 2 clases, hay que cambiar `Net(entradas=..., salidas=...)`
en `model.py`.

---

## 11. `entrenar_nodo.py`

**Responsabilidad:** orquestar todo: cargar datos, crear modelo, conectar al servidor, entrenar
por rondas, sincronizar matrices, evaluar y guardar.

| Líneas | Qué hace |
|---|---|
| 1-2 | `argparse`, `os`. |
| 4-6 | NumPy y PyTorch. |
| 8 | `import nnbridge`: módulo C++ compilado. |
| 9 | `from data import preparar`. |
| 10 | `from model import crear_modelo, extraer_matrices, actualizar_matrices`. |
| 13-17 | `obtener_batch(x, y, batch_size, paso)`: devuelve un batch circular. |
| 14 | `n = len(x)`. |
| 15 | `inicio = (paso * batch_size) % n`: desplazamiento circular. |
| 16 | `idx = (inicio + arange(batch_size)) % n`: índices, con vuelta al inicio. |
| 17 | Convierte a tensores `float32` e `int64`. |
| 20-25 | `evaluar(modelo, x, y)`: accuracy en test. |
| 21 | `modelo.eval()`. |
| 22 | `with torch.no_grad()`. |
| 23 | `logits = modelo(...)`. |
| 24 | `pred = argmax(logits, dim=1)`. |
| 25 | Devuelve `(pred == y).mean()`. |
| 28-95 | `main()`. |
| 29-45 | Define los argumentos de línea de comandos. |
| 30 | `--node`: obligatorio, 0/1/2. |
| 31 | `--nodos`: total, default 3. |
| 32 | `--host`: default `127.0.0.1`. |
| 33 | `--puerto`: default 5050. |
| 34 | `--rondas`: default 200. |
| 35 | `--batch-size`: default 32. |
| 36 | `--lr`: default 0.2. |
| 37 | `--seed`: default 42. |
| 38 | `--csv`: default `data/pima.csv`. |
| 39 | `--features`: índices separados por coma. |
| 40 | `--label`: default 8. |
| 41 | `--skiprows`: default 0. |
| 42 | `--salida`: carpeta de resultados. |
| 43 | `--pasos-locales`: default 5. |
| 44 | `--init`: npz con pesos iniciales compartidos. |
| 45 | `args = p.parse_args()`. |
| 47-49 | Convierte `--features` de string a lista de enteros. |
| 51 | Llama `preparar(...)` y obtiene shard y test. |
| 53 | Crea el modelo con la semilla. |
| 55-57 | Si `--init`, carga `w0`, `w1`, `w2` y actualiza el modelo. |
| 59 | `criterio = nn.CrossEntropyLoss()`. |
| 60 | `optimizador = torch.optim.SGD(..., lr=args.lr)`. |
| 62 | Imprime tamaño de shard, test y rondas. |
| 64 | `nnbridge.conectar(host, puerto, node, nodos, 60)`. |
| 65-88 | `try/finally` para garantizar `nnbridge.cerrar()`. |
| 66 | Bucle de rondas. |
| 67 | `modelo.train()`. |
| 69-76 | Bucle de pasos locales. |
| 70 | `paso_global = ronda * pasos_locales + paso_local`. |
| 71 | Obtiene batch. |
| 73 | `optimizador.zero_grad()`. |
| 74 | `perdida = criterio(modelo(xb), yb)`. |
| 75 | `perdida.backward()`. |
| 76 | `optimizador.step()`. |
| 78-82 | Sincronización con el servidor. |
| 79-81 | Para cada capa: enviar matriz y recibir media. |
| 82 | `actualizar_matrices(modelo, promedios)`. |
| 84-86 | Cada 10 rondas (y en la última), evalúa e imprime. |
| 87-88 | `finally: nnbridge.cerrar()`. |
| 90 | Crea la carpeta de salida. |
| 91 | Extrae los pesos finales. |
| 92-93 | Guarda `w0`, `w1`, `w2` en `resultados/nodo_X.npz`. |
| 95 | Imprime accuracy final y ruta. |
| 97-98 | `if __name__ == "__main__": main()`. |

### 11.1 El batch circular

`obtener_batch` no usa `DataLoader`; recorre el shard de forma circular:

```text
shard: [0, 1, 2, ..., 204]
paso 0, batch 32: índices 0..31
paso 1, batch 32: índices 32..63
...
si se acaba, vuelve al inicio con el módulo `% n`
```

Esto garantiza que siempre haya un batch completo, incluso si el shard no es múltiplo de
`batch_size`. La desventaja es que no hay shuffling por epoch; para este proyecto es suficiente.

### 11.2 Los 5 pasos locales

En cada ronda, el nodo hace 5 actualizaciones locales antes de sincronizar. Esto es importante
para la convergencia: si se sincronizara después de cada batch, el promedio sería muy ruidoso y
el modelo aprendería poco (fue el problema de la primera corrida con 50 rondas y 1 paso local).

### 11.3 Sincronización

```python
promedios = []
for capa, w in enumerate(extraer_matrices(modelo)):
    nnbridge.enviar_matriz(ronda, capa, w)
    promedios.append(nnbridge.recibir_matriz(ronda, capa))
actualizar_matrices(modelo, promedios)
```

- `extraer_matrices` devuelve las 3 matrices en orden.
- Se envía y se recibe **capa por capa**. No se envían las 3 y luego se reciben las 3, porque el
  servidor procesa en orden `(ronda, capa, nodo)`.
- `promedios` contiene las 3 matrices ya promediadas.
- `actualizar_matrices` las copia al modelo.

### 11.4 Guardado

El NPZ final contiene:

```text
w0: (6, 3) float64
w1: (6, 6) float64
w2: (2, 6) float64
```

Como el modelo tiene `bias=False`, estas 3 matrices son **todo** el modelo. Si los 3 NPZ son
iguales, los 3 modelos son idénticos.

### 11.5 Evaluación

`evaluar` usa el mismo test set en los 3 nodos (el split se calcula con el mismo `seed` y el mismo
CSV). Por eso los 3 deben imprimir la misma accuracy si los pesos son iguales.

---

## 12. `generar_init.py`

**Responsabilidad:** crear una sola vez `pesos_iniciales.npz` para que los 3 nodos arranquen con
los mismos pesos, sin depender del RNG de cada máquina.

| Líneas | Qué hace |
|---|---|
| 1 | Importa NumPy. |
| 3 | Importa `crear_modelo` y `extraer_matrices`. |
| 5 | Crea el modelo con `seed=42` y extrae las 3 matrices. |
| 6 | Guarda `w0`, `w1`, `w2` en `pesos_iniciales.npz`. |
| 7 | Imprime confirmación. |

Se ejecuta una vez:

```bash
python generar_init.py
```

Luego se copia `pesos_iniciales.npz` a las 3 máquinas y se pasa `--init pesos_iniciales.npz` a
cada nodo.

**Por qué es necesario:** aunque `torch.manual_seed(42)` produce los mismos pesos en la misma
versión de PyTorch, el RNG puede variar entre versiones o plataformas. El archivo compartido
elimina esa dependencia.

---

## 13. Pruebas incrementales

### 13.1 `test_step1.py` — pybind11 funciona

| Líneas | Qué hace |
|---|---|
| 1-2 | Importa NumPy y `nnbridge`. |
| 4 | Crea una matriz `2×3` con `np.arange`. |
| 5 | Llama `nnbridge.eco_matriz(a)`. |
| 7-9 | Verifica tipo, shape y contenido. |
| 11-14 | Imprime el resultado. |

Ejecutar:

```bash
python test_step1.py
```

Sirve para comprobar que el `.so`/`.pyd` se compiló y que pybind11 convierte NumPy correctamente.

### 13.2 `test_step2_cliente.py` — protocolo sin pybind11

Este cliente habla con el servidor usando **sockets puros de Python** y `struct.pack`, sin
`nnbridge`. Sirve para aislar problemas de red del binding C++.

| Líneas | Qué hace |
|---|---|
| 7-11 | Constantes `MAGIC`, `VERSION`, `DTYPE_FLOAT64` y formatos `struct`. |
| 10 | `HELLO = struct.Struct("<IHHHH")`: 12 bytes, little-endian. |
| 11 | `HEADER = struct.Struct("<IHHIHHIIQ")`: 32 bytes. |
| 14-21 | `recv_all`: recibe exactamente `n` bytes. |
| 24-28 | `enviar_matriz`: empaqueta cabecera y envía payload. |
| 31-36 | `recibir_matriz`: desempaqueta cabecera y reconstruye el array. |
| 39-66 | `main`: handshake, envía matrices con valor `node+1+capa`, verifica la media. |
| 42 | Puerto default **5000** (ojo: en macOS lo ocupa AirPlay; usar `--puerto 5050`). |
| 59 | Media esperada: `(num_nodos + 1) / 2 + capa`. |

Uso con 3 nodos:

```bash
./build/bin/servidor_central 5050 3 1 3
python test_step2_cliente.py --node 0 --nodos 3 --puerto 5050 &
python test_step2_cliente.py --node 1 --nodos 3 --puerto 5050 &
python test_step2_cliente.py --node 2 --nodos 3 --puerto 5050 &
wait
```

### 13.3 `test_step3_bridge.py` — bridge contra el servidor

Igual que el anterior, pero usando `nnbridge`. Es la prueba de la "Hoja 1": función C++ con socket
llamable desde Python.

| Líneas | Qué hace |
|---|---|
| 8-15 | Argumentos: host, puerto, node, nodos, rondas, capas. |
| 17 | `nnbridge.conectar(...)`. |
| 21-32 | Por ronda y capa: envía matriz y recibe media; valida. |
| 33-34 | `finally: nnbridge.cerrar()`. |

Defaults importantes:

- `--nodos` default **1**.
- `--rondas` default **1**.

Para probar 3 nodos hay que pasar `--nodos 3` y el mismo `--rondas` que el servidor.

### 13.4 `test_step4_model.py` — extracción, actualización y media

| Líneas | Qué hace |
|---|---|
| 7-13 | Verifica que el forward devuelva `(5, 2)` e imprime shapes. |
| 16-21 | Crea dos modelos con la misma semilla y compara parámetros. |
| 24-37 | Extrae matrices, verifica shapes/dtype, suma 1.0 y actualiza; comprueba el cambio. |
| 40-52 | Simula 3 modelos, calcula la media y la aplica; verifica. |
| 55-60 | Ejecuta todas las pruebas e imprime "Paso 4 OK". |

Shapes esperados: `(6,3)`, `(6,6)`, `(2,6)`, dtype `float64`.

---

## 14. Compilación y build

### 14.1 `CMakeLists.txt`

| Líneas | Qué hace |
|---|---|
| 1 | Versión mínima de CMake 3.18. |
| 2 | Proyecto `FinalProject`, lenguaje C++. |
| 4-5 | C++17 obligatorio. |
| 7 | Busca pybind11 (debe estar instalado en el venv). |
| 9 | Crea el módulo `nnbridge` desde `cpp/bridge.cpp`. |
| 10 | Agrega `cpp/` a los includes. |
| 11-14 | Deja el `.so`/`.pyd` en la raíz del proyecto. |
| 16 | Crea el ejecutable `servidor_central`. |
| 17 | Agrega `cpp/` a los includes. |
| 18-20 | Deja el ejecutable en `build/bin`. |
| 22-25 | En Windows enlaza `ws2_32` (Winsock) a ambos targets. |

**Mejora pendiente para Windows:** con Visual Studio (multi-config), el ejecutable y el `.pyd`
pueden quedar en `build/bin/Release` o `build/Release`. Si se necesita, agregar propiedades
`RUNTIME_OUTPUT_DIRECTORY_RELEASE`, `..._DEBUG`, etc., para cada configuración.

### 14.2 `scripts/build.sh`

| Líneas | Qué hace |
|---|---|
| 1 | Shebang de bash. |
| 2 | `set -euo pipefail`: aborta si falla un comando. |
| 4 | Se mueve a la raíz del proyecto. |
| 5 | Activa el venv. |
| 7-10 | Configura CMake con `compile_commands.json`, `pybind11_DIR` y `Python_EXECUTABLE`. |
| 12 | Compila. |

Uso:

```bash
./scripts/build.sh
```

Genera:

- `nnbridge.cpython-314-darwin.so` (macOS) o equivalente en Linux/Windows.
- `build/bin/servidor_central`.

### 14.3 `requirements.txt`

```text
pybind11==3.1.0
numpy>=2.0,<3
torch==2.15.0
```

Instalación:

```bash
python -m pip install -r requirements.txt
```

### 14.4 `.gitignore`

Ignora `.venv/`, `build/`, `__pycache__/`, `*.so`, `*.pyd`, `*.dylib`, `*.o`. No ignora el CSV,
los NPZ ni el código.

### 14.5 Compilación en Windows

Con Visual Studio Build Tools:

```bat
for /f "delims=" %i in ('python -m pybind11 --cmakedir') do set PYBIND11_DIR=%i
cmake -S . -B build -Dpybind11_DIR="%PYBIND11_DIR%" -DPython_EXECUTABLE="%CD%\.venv\Scripts\python.exe"
cmake --build build --config Release
```

Luego:

```bat
build\bin\servidor_central.exe 5050 3 200 3
```

y `python entrenar_nodo.py ...` desde la raíz.

### 14.6 Compilación en Linux

```bash
sudo apt install build-essential cmake python3-venv
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
./scripts/build.sh
```

---

## 15. Datos y artefactos

### 15.1 `data/pima.csv`

- 768 filas, 9 columnas, sin cabecera.
- Se usa para el clasificador binario de prueba.
- Debe ser idéntico en las 3 máquinas (mismo SHA256).

### 15.2 `pesos_iniciales.npz`

- Generado por `generar_init.py`.
- Contiene `w0: (6,3)`, `w1: (6,6)`, `w2: (2,6)`, todos `float64`.
- Se copia a las 3 máquinas.
- Se pasa con `--init pesos_iniciales.npz`.

### 15.3 `resultados/nodo_X.npz`

- Generado al final de `entrenar_nodo.py`.
- Contiene los pesos finales del nodo X.
- Si los 3 archivos tienen los mismos arrays, los modelos son idénticos.

Verificación:

```bash
python - <<'EOF'
import numpy as np

base = np.load("resultados/nodo_0.npz")
for n in (1, 2):
    otro = np.load(f"resultados/nodo_{n}.npz")
    print(f"nodo {n}: identicos ->",
          all(np.allclose(base[k], otro[k]) for k in base.files))
EOF
```

### 15.4 `nnbridge.*.so` / `nnbridge.*.pyd`

- Módulo compilado de C++.
- Debe estar en la raíz del proyecto o en `PYTHONPATH`.
- No se comparte entre sistemas operativos: cada máquina compila el suyo.

---

## 16. Ejecución localhost y LAN

### 16.1 Localhost (3 terminales)

Terminal 1:

```bash
./build/bin/servidor_central 5050 3 200 3
```

Terminal 2, 3 y 4 (cambiando `--node`):

```bash
python entrenar_nodo.py --node 0 --nodos 3 --host 127.0.0.1 --puerto 5050 \
  --rondas 200 --pasos-locales 5 --lr 0.2 --batch-size 32 \
  --init pesos_iniciales.npz
```

Resultado esperado:

- Servidor: `ronda 200/200 OK` y `listo`.
- Nodos: `accuracy final=0.7532`.
- Verificación: `identicos -> True` para nodos 1 y 2.

### 16.2 LAN

1. Copiar a las 3 máquinas: código, `data/pima.csv`, `pesos_iniciales.npz`.
2. Compilar en cada una.
3. Verificar el hash del CSV.
4. Permitir el puerto 5050 en el firewall.
5. Servidor en una máquina.
6. Un cliente por máquina, con `--host <IP_SERVIDOR>`.

Reglas:

- Exactamente 3 clientes.
- Un `node_id` distinto por cliente.
- Mismo `--rondas` en servidor y clientes.
- Todos corriendo al mismo tiempo.
- No relanzar clientes después de que el servidor imprima `listo`.

### 16.3 Prueba corta antes de entrenar

Servidor:

```bash
./build/bin/servidor_central 5050 3 1 3
```

Clientes:

```bash
python test_step3_bridge.py --node 0 --nodos 3 --host 127.0.0.1 --puerto 5050 --rondas 1
```

Repetir para `--node 1` y `--node 2`, en terminales separadas o con `&` y `wait`.

---

## 17. Protocolo y secuencia

### 17.1 Handshake

1. Cliente crea socket TCP y conecta.
2. Cliente envía `Hello` con `node_id` y `num_nodos`.
3. Servidor valida y guarda el socket en `nodos[node_id]`.
4. Cuando tiene `num_nodos` únicos, empieza el entrenamiento.

### 17.2 Mensaje de matriz

```text
[ MsgHeader: 32 bytes ][ payload: filas*columnas*8 bytes ]
```

### 17.3 Orden de capas

| Capa | Matriz | Shape |
|---|---|---|
| 0 | `fc1.weight` | `(6, 3)` |
| 1 | `fc2.weight` | `(6, 6)` |
| 2 | `fc3.weight` | `(2, 6)` |

### 17.4 Orden de sincronización

Por ronda:

```text
capa 0: todos envían, servidor promedia, todos reciben
capa 1: todos envían, servidor promedia, todos reciben
capa 2: todos envían, servidor promedia, todos reciben
```

No se puede adelantar una capa ni cambiar el orden.

### 17.5 Byte order

Los structs están empaquetados con `#pragma pack(push, 1)`. Los enteros se escriben en el orden
nativo (little-endian en x86/ARM). Las 3 máquinas del proyecto son little-endian, así que no se
hace conversión explícita. Si se usara una arquitectura big-endian, habría que convertir con
`htonl`/`ntohl`.

---

## 18. Errores y depuración

### 18.1 Errores de compilación

| Error | Causa | Solución |
|---|---|---|
| `pybind11/pybind11.h file not found` | Cursor/clangd no conoce los includes | Compilar con CMake; generar `compile_commands.json`; no compilar con `clang++` a mano. |
| `python: command not found` | venv no activado | `source .venv/bin/activate`. |
| `Could not find pybind11` | `pybind11_DIR` vacío | Activar venv y pasar `-Dpybind11_DIR="$(python -m pybind11 --cmakedir)"`. |
| `Python_EXECUTABLE=python not found` | CMake cache envenenada | Borrar `build/` y reconfigurar. |
| `no type named socket_t in namespace red` | `using` fuera del namespace | Mover `using socket_t` y `SOCKET_INVALIDO` dentro de `namespace red`. |
| `ws2_32` no encontrado (Windows) | Falta enlazar Winsock | `target_link_libraries(... ws2_32)`. |

### 18.2 Errores de ejecución

| Error | Causa | Solución |
|---|---|---|
| `bind: Address already in use` | Otro proceso usa el puerto; en macOS el 5000 es AirPlay | Usar 5050; `lsof -nP -iTCP:5050 -sTCP:LISTEN`. |
| `no se pudo conectar` | Servidor no corriendo, IP/puerto mal | Arrancar servidor; revisar `--host`/`--puerto`; firewall. |
| `node_id duplicado` | Dos clientes con el mismo `--node` | Exactamente un cliente por nodo. |
| `recv_all: conexion cerrada` | Un nodo se cayó o cerró | Detener todo y reiniciar limpio. |
| `recv_all: Resource temporarily unavailable` | Timeout de socket | Revisar que los 3 nodos estén enviando; aumentar timeout si hace falta. |
| `ronda/capa inesperada` | Servidor y clientes con distinto `--rondas` o desfase | Mismo `--rondas`; no mezclar ejecuciones. |
| `shapes distintas entre nodos` | Nodos con distinto modelo, features o CSV | Mismos argumentos, mismo `model.py`, mismo CSV. |
| `ModuleNotFoundError: nnbridge` | No se compiló o no se ejecuta desde la raíz | `./scripts/build.sh`; ejecutar desde `FinalProject/`. |
| accuracy baja (~0.54) | Muy pocos pasos locales o bias local | Usar `bias=False`, `--pasos-locales 5`, 200 rondas, split estratificado. |

### 18.3 Diagnóstico de procesos

```bash
ps aux | grep -E "servidor_central|entrenar_nodo|test_step3_bridge"
lsof -nP -iTCP:5050 -sTCP:LISTEN
```

Si hay procesos viejos, ciérralos con `Ctrl+C` o `kill <PID>`.

### 18.4 Diagnóstico de pesos

```bash
python - <<'EOF'
import numpy as np

base = np.load("resultados/nodo_0.npz")
for n in (1, 2):
    otro = np.load(f"resultados/nodo_{n}.npz")
    print(n, all(np.allclose(base[k], otro[k]) for k in base.files))
EOF
```

---

## 19. Guía del 40%: modificaciones probables

El profesor pedirá un cambio pequeño, individual, y evaluará si cada alumno entiende su código.
La clave es saber **dónde** está el cambio y **qué más** hay que ajustar para que siga funcionando.

### 19.1 Mapa rápido de cambios

| Cambio | Archivo(s) | Qué tocar |
|---|---|---|
| 6 → 8 neuronas ocultas | `model.py` | `Net.__init__`, regenerar init, actualizar test. |
| ReLU → Tanh/Sigmoid | `model.py` | `forward`. |
| Agregar una capa | `model.py`, `entrenar_nodo.py`, `generar_init.py`, servidor | `fc4`, `matrices_de_pesos`, `--init`, `--capas 4`. |
| Promediar bias | `model.py`, `generar_init.py`, `entrenar_nodo.py`, servidor | bias como matrices `1×N`, `--capas 6`. |
| Media ponderada | `servidor_central.cpp` | bucle de media. |
| Mediana o máximo | `servidor_central.cpp` | bucle de media. |
| Cambiar features/label | CLI o `data.py` | `--features`, `--label`, `--skiprows`. |
| Cambiar lr/batch/rondas | CLI | `--lr`, `--batch-size`, `--rondas`, `--pasos-locales`. |
| Cambiar puerto | CLI | `--puerto`. |
| 4 nodos | CLI + servidor | `--nodos 4`, 4 clientes. |
| Guardar/cargar modelo | `entrenar_nodo.py` | ya guarda NPZ; agregar carga. |
| float32 en el cable | `protocol.hpp`, `bridge.cpp`, `model.py` | nuevo dtype y `py::array_t<float>`. |

### 19.2 Receta A: cambiar 6 → 8 neuronas

1. En `model.py`, línea 8:
   ```python
   def __init__(self, entradas=3, ocultas=8, salidas=2):
   ```
2. `crear_modelo` sigue usando `Net()` con el nuevo default.
3. Los shapes cambian:
   - `fc1.weight`: `(8, 3)`
   - `fc2.weight`: `(8, 8)`
   - `fc3.weight`: `(2, 8)`
4. Regenerar `pesos_iniciales.npz`:
   ```bash
   python generar_init.py
   ```
5. Actualizar `test_step4_model.py` línea 28 con los nuevos shapes.
6. El servidor no necesita cambios: solo compara shapes entre nodos.
7. Probar:
   ```bash
   python test_step4_model.py
   ./build/bin/servidor_central 5050 3 10 3
   python entrenar_nodo.py --node 0 --rondas 10 ...
   ```

**Lo que evalúa el profesor:** que sepas que el cambio afecta los shapes y que hay que regenerar
los pesos iniciales.

### 19.3 Receta B: cambiar ReLU por Tanh

1. En `model.py`, líneas 15-16:
   ```python
   x = F.tanh(self.fc1(x))
   x = F.tanh(self.fc2(x))
   ```
2. No cambian shapes ni protocolo.
3. Probar 10 rondas; si la pérdida no baja, ajustar `--lr`.
4. Explicar: Tanh satura en valores grandes; ReLU no. Por eso puede requerir otro lr.

### 19.4 Receta C: agregar una capa oculta

1. En `model.py`, línea 12, agregar:
   ```python
   self.fc4 = nn.Linear(ocultas, salidas, bias=False)
   ```
   y cambiar `fc3` para que sea `nn.Linear(ocultas, ocultas, bias=False)`.
2. En `forward`, línea 17:
   ```python
   x = F.relu(self.fc3(x))
   return self.fc4(x)
   ```
3. En `matrices_de_pesos`, línea 28, agregar `modelo.fc4.weight`.
4. En `generar_init.py`, guardar `w3`:
   ```python
   np.savez("pesos_iniciales.npz", w0=pesos[0], w1=pesos[1], w2=pesos[2], w3=pesos[3])
   ```
5. En `entrenar_nodo.py`, líneas 55-57, cargar `w3` si existe:
   ```python
   claves = ["w0", "w1", "w2", "w3"]
   actualizar_matrices(modelo, [datos[k] for k in claves if k in datos.files])
   ```
6. Servidor y clientes con `--capas 4`.
7. Actualizar `test_step4_model.py` para 4 matrices.
8. Regenerar init y probar 10 rondas.

**Trampa:** el código actual hardcodea `w0`, `w1`, `w2` en `generar_init.py` y en `--init`.
Si solo agregas `fc4`, el `--init` cargaría 3 matrices y `actualizar_matrices` fallaría porque
espera 4.

### 19.5 Receta D: promediar también los bias

1. En `model.py`, líneas 10-12: `bias=True`.
2. `matrices_de_pesos` debe devolver 6 tensores o `extraer_matrices` debe incluir bias como
   matrices `1×N`:
   ```python
   def extraer_matrices(modelo):
       pesos = [modelo.fc1.weight, modelo.fc2.weight, modelo.fc3.weight]
       biases = [modelo.fc1.bias, modelo.fc2.bias, modelo.fc3.bias]
       return [p.detach().cpu().numpy().astype(np.float64, copy=True) for p in pesos + biases]
   ```
   Pero los bias son 1-D; el protocolo espera 2-D. Hay que hacer `reshape(1, -1)` al extraer y
   `reshape(-1)` al actualizar.
3. `actualizar_matrices` debe distinguir pesos 2-D y bias 1-D.
4. `generar_init.py` debe guardar 6 matrices (`w0..w2`, `b0..b2`).
5. `entrenar_nodo.py --init` debe cargar las 6.
6. Servidor y clientes con `--capas 6`.
7. Probar 10 rondas.

**Alternativa más simple:** dejar `bias=False` y explicar que así las 3 matrices son todo el
modelo. Si el profesor pide promediar bias, la receta anterior es la respuesta completa.

### 19.6 Receta E: media ponderada

1. En `servidor_central.cpp`, líneas 107-111, reemplazar la media simple por:
   ```cpp
   std::vector<double> pesos_nodos = {0.334, 0.333, 0.333}; // o calcular con tamaños de shard
   std::vector<double> media(filas * columnas, 0.0);
   double suma_pesos = 0.0;
   for (int i = 0; i < num_nodos; ++i) {
       suma_pesos += pesos_nodos[i];
       for (size_t k = 0; k < media.size(); ++k)
           media[k] += pesos_nodos[i] * matrices[i][k];
   }
   for (double& v : media) v /= suma_pesos;
   ```
2. Si los pesos deben venir de los clientes, agregar un campo al `Hello` o un mensaje nuevo.
3. Probar con `test_step2_cliente.py` o una corrida corta.
4. Explicar la diferencia: la media simple da el mismo peso a cada nodo; la ponderada da más peso
   a los nodos con más datos.

### 19.7 Receta F: mediana o máximo

En `servidor_central.cpp`, líneas 107-111, para cada celda `k`:

```cpp
std::vector<double> valores;
for (int i = 0; i < num_nodos; ++i)
    valores.push_back(matrices[i][k]);
std::sort(valores.begin(), valores.end());
media[k] = valores[valores.size() / 2]; // mediana
// o: media[k] = valores.back(); // máximo
```

Incluir `<algorithm>`. No cambia el protocolo.

### 19.8 Receta G: cambiar dataset o features

1. Si el nuevo CSV tiene 3 features y 2 clases, solo CLI:
   ```bash
   python entrenar_nodo.py --csv data/otro.csv --features 0,1,2 --label 3 --skiprows 1
   ```
2. Si tiene otra cantidad de features, cambiar `Net(entradas=N)` en `model.py` o el default.
3. Si tiene otra cantidad de clases, cambiar `salidas` y las etiquetas.
4. Los 3 nodos deben usar el mismo CSV y los mismos argumentos.

### 19.9 Receta H: cambiar hiperparámetros

Solo CLI:

```bash
--rondas 100
--pasos-locales 10
--lr 0.1
--batch-size 64
```

El servidor debe usar el mismo `--rondas` que los clientes. No hay que recompilar.

### 19.10 Receta I: 4 nodos

1. Servidor: `./build/bin/servidor_central 5050 4 200 3`.
2. Clientes: `--nodos 4`, con `--node 0`, `1`, `2`, `3`.
3. `data.py` reparte el train en 4 shards automáticamente.
4. El servidor divide la media entre 4.
5. No hay cambios de código; solo argumentos y lanzar 4 clientes.

### 19.11 Receta J: cambiar el puerto

```bash
./build/bin/servidor_central 6000 3 200 3
python entrenar_nodo.py --puerto 6000 ...
```

No hay que recompilar. Recordar firewall y que el 5000 está ocupado por AirPlay en macOS.

### 19.12 Receta K: float32 en el cable

1. En `protocol.hpp`, agregar `DTYPE_FLOAT32 = 1` y cambiar `enviar_matriz`/`recibir_matriz` para
   usar `float` y `nbytes = filas * columnas * 4`.
2. En `bridge.cpp`, cambiar `py::array_t<double>` por `py::array_t<float>` y los casts.
3. En `model.py`, extraer `float32` en vez de `float64`.
4. El servidor debe usar `std::vector<float>` y `float` en la media.
5. Probar. Es un cambio más grande; el profesor probablemente no lo pida.

### 19.13 Cómo practicar

1. Cada integrante elige una receta.
2. La aplica en su máquina.
3. Corre una prueba corta (10 rondas).
4. Verifica que compila, que los 3 nodos sincronizan y que la accuracy es razonable.
5. Reverte el cambio.
6. Cronometra: el profesor pedirá algo "chiquito pero complicado"; hay que saber ubicarlo rápido.

---

## 20. Glosario

### 20.1 Funciones C++

| Función | Archivo | Qué hace |
|---|---|---|
| `red::inicializar_sockets()` | `socket_utils.hpp` | Inicializa Winsock o ignora `SIGPIPE`. |
| `red::cerrar_socket(s)` | `socket_utils.hpp` | Cierra un socket de forma portable. |
| `red::error_socket()` | `socket_utils.hpp` | Devuelve el último error del sistema. |
| `red::configurar_timeout(s, seg)` | `socket_utils.hpp` | Configura timeouts de envío/recepción. |
| `red::send_all(s, datos, n)` | `protocol.hpp` | Envía exactamente `n` bytes. |
| `red::recv_all(s, datos, n)` | `protocol.hpp` | Recibe exactamente `n` bytes. |
| `red::enviar_hello(s, node, n)` | `protocol.hpp` | Envía el handshake. |
| `red::recibir_hello(s)` | `protocol.hpp` | Recibe y valida el handshake. |
| `red::enviar_matriz(...)` | `protocol.hpp` | Envía cabecera + payload de una matriz. |
| `red::recibir_matriz(...)` | `protocol.hpp` | Recibe cabecera + payload. |
| `crear_listen_socket(puerto)` | `servidor_central.cpp` | Crea, bindea y pone en escucha el socket. |
| `aceptar_con_timeout(s, seg)` | `servidor_central.cpp` | Acepta una conexión con timeout. |
| `conectar_tcp(host, puerto)` | `bridge.cpp` | Resuelve y conecta un socket cliente. |
| `eco_matriz(m)` | `bridge.cpp` | Copia una matriz (prueba). |
| `py_conectar(...)` | `bridge.cpp` | Conecta y envía Hello desde Python. |
| `py_cerrar()` | `bridge.cpp` | Cierra la conexión. |
| `py_enviar_matriz(...)` | `bridge.cpp` | Envía una matriz NumPy. |
| `py_recibir_matriz(...)` | `bridge.cpp` | Recibe una matriz y la devuelve como NumPy. |
| `py_sincronizar_matriz(...)` | `bridge.cpp` | Envía y recibe en una llamada. |

### 20.2 Funciones Python

| Función | Archivo | Qué hace |
|---|---|---|
| `Net.__init__` | `model.py` | Define las 3 capas. |
| `Net.forward` | `model.py` | Forward: ReLU, ReLU, lineal. |
| `crear_modelo` | `model.py` | Crea la red con semilla. |
| `matrices_de_pesos` | `model.py` | Devuelve las 3 matrices en orden. |
| `extraer_matrices` | `model.py` | Convierte pesos a NumPy float64. |
| `actualizar_matrices` | `model.py` | Copia matrices al modelo. |
| `resumen_modelo` | `model.py` | Imprime shapes/dtype/device. |
| `_leer_csv` | `data.py` | Lee CSV local o descarga Pima. |
| `datos_sinteticos` | `data.py` | Genera datos de respaldo. |
| `cargar_datos` | `data.py` | Selecciona features y etiqueta. |
| `preparar` | `data.py` | Split, normalización y shard. |
| `obtener_batch` | `entrenar_nodo.py` | Batch circular. |
| `evaluar` | `entrenar_nodo.py` | Accuracy en test. |
| `main` | `entrenar_nodo.py` | Orquesta todo. |
| `test_*` | tests | Pruebas incrementales. |

### 20.3 Variables y constantes

| Nombre | Dónde | Significado |
|---|---|---|
| `MAGIC` | `protocol.hpp` | `0x4E4E5731`, identifica el protocolo. |
| `VERSION` | `protocol.hpp` | Versión 1. |
| `DTYPE_FLOAT64` | `protocol.hpp` | Código 0 para `double`. |
| `Hello` | `protocol.hpp` | Estructura de handshake. |
| `MsgHeader` | `protocol.hpp` | Cabecera de matriz. |
| `nbytes` | `protocol.hpp` | Tamaño del payload en bytes. |
| `filas`, `columnas` | protocolo/servidor | Shape de la matriz. |
| `ronda` | protocolo/servidor | Número de ronda. |
| `capa` | protocolo/servidor | Índice de capa (0, 1, 2). |
| `node_id` | protocolo/bridge | Índice del nodo. |
| `num_nodos` | protocolo/servidor | Total de nodos. |
| `g_socket` | `bridge.cpp` | Socket global del proceso. |
| `g_node_id` | `bridge.cpp` | Node id global. |
| `x_shard`, `y_shard` | `entrenar_nodo.py` | Datos del shard local. |
| `x_test`, `y_test` | `entrenar_nodo.py` | Datos de test. |
| `mu`, `sigma` | `data.py` | Media y desviación para normalizar. |
| `train_idx`, `test_idx` | `data.py` | Índices del split. |
| `partes` | `data.py` | Lista de shards. |
| `logits` | `entrenar_nodo.py` | Salida de la red antes de softmax. |
| `pred` | `entrenar_nodo.py` | Clase predicha. |
| `perdida` | `entrenar_nodo.py` | Loss del último paso local. |
| `promedios` | `entrenar_nodo.py` | Matrices devueltas por el servidor. |
| `pesos` | `entrenar_nodo.py` | Pesos finales extraídos. |
| `w0`, `w1`, `w2` | NPZ | Matrices `fc1`, `fc2`, `fc3`. |

### 20.4 Términos de redes

| Término | Significado |
|---|---|
| **Socket** | Extremo de una conexión de red identificado por IP y puerto. |
| **TCP** | Protocolo orientado a conexión y confiable; entrega bytes en orden. |
| **Handshake** | Intercambio inicial para identificarse (aquí `Hello`). |
| **Frame/cabecera** | Metadatos que indican tamaño y tipo del mensaje. |
| **Payload** | Datos útiles del mensaje (la matriz). |
| **Timeout** | Tiempo máximo de espera antes de fallar. |
| **bind** | Asociar un socket a una IP/puerto local. |
| **listen** | Poner un socket en modo escucha. |
| **accept** | Aceptar una conexión entrante. |
| **connect** | Iniciar una conexión a un servidor. |
| **send/recv** | Enviar/recibir bytes. |
| **`SO_REUSEADDR`** | Permite reutilizar el puerto al reiniciar. |
| **`SO_RCVTIMEO`/`SO_SNDTIMEO`** | Timeouts de recepción/emisión. |
| **`INADDR_ANY`** | Escuchar en todas las interfaces. |
| **`htons`/`htonl`** | Convertir enteros al orden de red. |
| **`getaddrinfo`** | Resolver nombre de host y puerto. |
| **Little-endian** | Orden de bytes de x86/ARM. |
| **`SIGPIPE`** | Señal que mata el proceso al escribir a un socket cerrado. |
| **Winsock** | API de sockets de Windows. |
| **POSIX** | API de sockets de macOS/Linux. |

### 20.5 Términos de ML

| Término | Significado |
|---|---|
| **Peso (weight)** | Parámetro aprendido de una capa lineal. |
| **Bias** | Término independiente de una capa; aquí no se usa. |
| **Logits** | Salida cruda de la última capa, antes de softmax. |
| **Forward** | Cálculo de la salida. |
| **Backward** | Cálculo de gradientes. |
| **Gradiente** | Derivada de la pérdida respecto a un parámetro. |
| **Optimizador** | Algoritmo que actualiza parámetros (aquí SGD). |
| **SGD** | Descenso de gradiente estocástico. |
| **CrossEntropyLoss** | Pérdida para clasificación con logits. |
| **ReLU** | `max(0, x)`. |
| **Tanh/Sigmoid** | Activaciones alternativas. |
| **Batch** | Grupo de muestras. |
| **Epoch** | Una pasada completa por el dataset. |
| **Shard** | Partición del dataset asignada a un nodo. |
| **Ronda** | Grupo de pasos locales + sincronización. |
| **Federated Averaging** | Promedio de parámetros entre nodos. |
| **Accuracy** | Proporción de predicciones correctas. |
| **Precisión** | TP / (TP + FP). |
| **Recall** | TP / (TP + FN). |
| **Especificidad** | TN / (TN + FP). |
| **Matriz de confusión** | Tabla real vs predicho. |
| **Estandarización** | `(x - mu) / sigma`. |
| **Split estratificado** | Mantiene proporción de clases. |
| **`detach()`** | Separa un tensor del grafo. |
| **`no_grad()`** | No registra operaciones para autograd. |
| **`state_dict`** | Diccionario de parámetros de un modelo. |

### 20.6 pybind11 y CMake

| Término | Significado |
|---|---|
| **pybind11** | Librería para exponer C++ a Python. |
| **`PYBIND11_MODULE`** | Macro que define el módulo. |
| **`py::array_t<double>`** | Tipo de array NumPy en C++. |
| **`py::buffer_info`** | Shape, strides, dtype y puntero. |
| **`c_style`** | Memoria contigua en orden C. |
| **`forcecast`** | Convierte el dtype automáticamente. |
| **`py::arg`** | Nombre de argumento en Python. |
| **GIL** | Global Interpreter Lock de Python. |
| **CMake** | Sistema de build. |
| **`pybind11_add_module`** | Función CMake que crea el módulo. |
| **`target_include_directories`** | Agrega rutas de includes. |
| **`ws2_32`** | Librería de Winsock en Windows. |

---

## 21. Preguntas de defensa

1. **¿Dónde se extraen las matrices?**  
   En `model.py`, `extraer_matrices`, líneas 31-35. Convierte `fc1.weight`, `fc2.weight` y
   `fc3.weight` a NumPy `float64`.

2. **¿Dónde se actualizan?**  
   En `model.py`, `actualizar_matrices`, líneas 38-51. Valida shapes y usa `param.copy_` dentro
   de `torch.no_grad()`.

3. **¿Dónde se calcula la media?**  
   En `cpp/servidor_central.cpp`, líneas 107-111. Suma las matrices de los 3 nodos y divide cada
   celda entre 3.

4. **¿Cómo se comunican Python y C++?**  
   Con pybind11. `cpp/bridge.cpp` define el módulo `nnbridge` y expone `conectar`,
   `enviar_matriz`, `recibir_matriz`, `sincronizar_matriz` y `cerrar`.

5. **¿Cómo se comunican los nodos con el servidor?**  
   Por sockets TCP. El protocolo está en `cpp/protocol.hpp`: `Hello` y `MsgHeader`.

6. **¿Por qué hay 3 matrices?**  
   Porque la red tiene 3 capas lineales y `bias=False`. Los shapes son `(6,3)`, `(6,6)` y `(2,6)`.

7. **¿Por qué `bias=False`?**  
   Para que las 3 matrices sean todo el modelo y el promedio deje los 3 modelos idénticos. Si
   hubiera bias, quedarían distintos en cada nodo porque no se promedian.

8. **¿Qué es un shard?**  
   La parte del dataset que entrena cada nodo. `data.py` reparte el train con `np.array_split`.

9. **¿Por qué se usa un split estratificado?**  
   Para mantener la proporción de positivos/negativos en train y test. Sin eso, el test puede
   quedar desbalanceado y la accuracy engañosa.

10. **¿Por qué se normaliza?**  
    Para que las features tengan media 0 y desviación 1, lo que estabiliza el entrenamiento.

11. **¿Por qué se usa `--init`?**  
    Para que los 3 nodos arranquen con los mismos pesos, sin depender del RNG de cada máquina.

12. **¿Qué pasa si un nodo no envía?**  
    El servidor espera con timeout de 60 s; si no llega, `recv_all` falla y el servidor termina
    con error.

13. **¿Qué pasa si hay dos clientes con el mismo `node_id`?**  
    El servidor rechaza el duplicado y el cliente falla. Debe haber exactamente un cliente por
    nodo.

14. **¿Por qué el servidor usa orden fijo?**  
    Para evitar hilos y condiciones de carrera. Es suficiente para 3 nodos y hace el flujo
    determinista.

15. **¿Por qué se envían `filas`, `columnas` y `nbytes`?**  
    Porque TCP es un flujo de bytes; el receptor necesita saber cuánto leer y qué shape tiene la
    matriz.

16. **¿Por qué `send_all` y `recv_all`?**  
    Porque `send`/`recv` pueden transferir menos bytes de los pedidos; hay que repetir hasta
    completar.

17. **¿Por qué `float64` en el cable?**  
    Para evitar pérdida de precisión en la media y usar `py::array_t<double>`. PyTorch usa
    `float32`, pero se convierte al extraer y al actualizar.

18. **¿Cómo se garantiza que los modelos quedan iguales?**  
    Los 3 usan el mismo `--init`, el servidor promedia las mismas 3 matrices y todos actualizan
    con la misma media. Al final, los NPZ son iguales.

19. **¿Qué cambiarías para 4 nodos?**  
    Servidor `5050 4 200 3`, clientes `--nodos 4` y un `--node` distinto cada uno. `data.py`
    reparte en 4 shards automáticamente.

20. **¿Cómo promediarías los bias?**  
    Activando `bias=True`, extrayendo los 3 bias como matrices `1×N`, enviándolos como capas 3,
    4 y 5, y usando `--capas 6`. Hay que actualizar `generar_init.py` y `--init`.

21. **¿Cómo cambiarías la media por mediana?**  
    En `servidor_central.cpp`, para cada celda, juntar los 3 valores, ordenarlos y tomar el del
    medio.

22. **¿Cómo cambiarías la media por media ponderada?**  
    Multiplicar cada matriz por un peso (por ejemplo, tamaño de shard), sumar y dividir entre la
    suma de pesos.

23. **¿Por qué no se usa GPU?**  
    La red es diminuta (66 parámetros) y el dataset es pequeño. La parte costosa es la
    comunicación. Además, CPU evita diferencias de precisión entre MPS/CUDA y CPU.

24. **¿Qué hace `torch.manual_seed`?**  
    Fija la semilla del RNG para que la inicialización sea reproducible. Aun así, se usa
    `pesos_iniciales.npz` para independizarse de la versión de PyTorch.

25. **¿Qué hace `detach()`?**  
    Separa el tensor del grafo de autograd para poder convertirlo a NumPy sin arrastrar
    gradientes.

26. **¿Qué hace `param.copy_`?**  
    Copia los valores in-place en el parámetro, manteniendo la referencia que usa el optimizador.

27. **¿Qué es el GIL?**  
    El bloqueo global del intérprete de Python. Las llamadas a `nnbridge` lo mantienen mientras
    esperan en el socket; en este proyecto no hay hilos Python, así que no afecta.

28. **¿Qué pasa si el servidor no está corriendo?**  
    `conectar` falla con "no se pudo conectar". Hay que arrancar el servidor primero.

29. **¿Qué pasa si el puerto 5050 está ocupado?**  
    `bind: Address already in use`. Hay que cerrar el proceso que lo usa o cambiar de puerto.

30. **¿Qué evidencia tienes de que funciona?**  
    `test_step3_bridge.py` (3 nodos, medias 2.0/3.0/4.0), `test_step4_model.py`, la corrida de
    200 rondas con accuracy 0.7532 y la verificación de NPZ idénticos.

---

## 22. Checklist final

### Antes de la entrega

- [ ] El proyecto compila en macOS con `./scripts/build.sh`.
- [ ] Los tests `test_step1.py`, `test_step3_bridge.py` y `test_step4_model.py` pasan.
- [ ] La corrida local de 3 nodos termina con accuracy ~0.7532.
- [ ] Los 3 `resultados/nodo_X.npz` son idénticos.
- [ ] `README.md` explica compilación, ejecución y resultados.
- [ ] `requirements.txt` tiene las versiones correctas.
- [ ] `pesos_iniciales.npz` y `data/pima.csv` están incluidos.
- [ ] No se incluyen `.venv/`, `build/` ni `__pycache__/`.
- [ ] Cada integrante puede compilar y correr en su máquina.
- [ ] La prueba LAN está hecha o documentada como pendiente.

### Para el 40%

- [ ] Cada integrante sabe dónde están las 3 matrices (`model.py`).
- [ ] Cada integrante sabe dónde se calcula la media (`servidor_central.cpp`).
- [ ] Cada integrante sabe dónde se exponen las funciones (`bridge.cpp`).
- [ ] Cada integrante sabe cambiar `ocultas=6` a otro valor.
- [ ] Cada integrante sabe cambiar ReLU por otra activación.
- [ ] Cada integrante sabe agregar una capa y qué más actualizar.
- [ ] Cada integrante sabe promediar bias (o explicar por qué no se usan).
- [ ] Cada integrante sabe cambiar la media por mediana o media ponderada.
- [ ] Cada integrante practicó al menos un cambio y lo revirtió.
- [ ] Cada integrante puede explicar el flujo de una ronda sin mirar el código.

### Frase para recordar

> Python entrena y extrae las 3 matrices; C++ las transporta por TCP; el servidor calcula la
> media celda por celda; Python actualiza el modelo con esa media; se repite hasta que los 3
> modelos quedan idénticos.
