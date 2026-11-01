#pragma once

#include "socket_utils.hpp"

#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <vector>

namespace red {

constexpr uint32_t MAGIC = 0x4E4E5731u; // "NNW1"
constexpr uint16_t VERSION = 1;
constexpr uint16_t DTYPE_FLOAT64 = 0;

#pragma pack(push, 1)
struct Hello {
    uint32_t magic;
    uint16_t version;
    uint16_t node_id;
    uint16_t num_nodos;
    uint16_t reservado;
};

struct MsgHeader {
    uint32_t magic;
    uint16_t version;
    uint16_t node_id;
    uint32_t ronda;
    uint16_t capa;
    uint16_t dtype;
    uint32_t filas;
    uint32_t columnas;
    uint64_t nbytes;
};
#pragma pack(pop)

static_assert(sizeof(Hello) == 12, "Hello debe medir 12 bytes");
static_assert(sizeof(MsgHeader) == 32, "MsgHeader debe medir 32 bytes");

inline void send_all(socket_t s, const void* datos, size_t n) {
    const char* p = static_cast<const char*>(datos);
    size_t enviado = 0;
    while (enviado < n) {
#ifdef _WIN32
        int r = ::send(s, p + enviado, static_cast<int>(n - enviado), 0);
#else
        ssize_t r = ::send(s, p + enviado, n - enviado, 0);
#endif
        if (r <= 0) throw std::runtime_error("send_all: " + error_socket());
        enviado += static_cast<size_t>(r);
    }
}

inline void recv_all(socket_t s, void* datos, size_t n) {
    char* p = static_cast<char*>(datos);
    size_t recibido = 0;
    while (recibido < n) {
#ifdef _WIN32
        int r = ::recv(s, p + recibido, static_cast<int>(n - recibido), 0);
#else
        ssize_t r = ::recv(s, p + recibido, n - recibido, 0);
#endif
        if (r == 0) throw std::runtime_error("recv_all: conexion cerrada");
        if (r < 0) throw std::runtime_error("recv_all: " + error_socket());
        recibido += static_cast<size_t>(r);
    }
}

inline void enviar_hello(socket_t s, int node_id, int num_nodos) {
    Hello h{MAGIC, VERSION, static_cast<uint16_t>(node_id), static_cast<uint16_t>(num_nodos), 0};
    send_all(s, &h, sizeof(h));
}

inline Hello recibir_hello(socket_t s) {
    Hello h{};
    recv_all(s, &h, sizeof(h));
    if (h.magic != MAGIC) throw std::runtime_error("hello: magic invalido");
    if (h.version != VERSION) throw std::runtime_error("hello: version invalida");
    return h;
}

inline void enviar_matriz(socket_t s, int node_id, uint32_t ronda, uint16_t capa, uint32_t filas, uint32_t columnas, const double* datos) {
    MsgHeader h{MAGIC, VERSION, static_cast<uint16_t>(node_id), ronda, capa, DTYPE_FLOAT64, filas, columnas, static_cast<uint64_t>(filas) * columnas * sizeof(double)};
    send_all(s, &h, sizeof(h));
    send_all(s, datos, static_cast<size_t>(h.nbytes));
}

inline MsgHeader recibir_matriz(socket_t s, std::vector<double>& datos) {
    MsgHeader h{};
    recv_all(s, &h, sizeof(h));
    if (h.magic != MAGIC) throw std::runtime_error("matriz: magic invalido");
    if (h.version != VERSION) throw std::runtime_error("matriz: version invalida");
    if (h.dtype != DTYPE_FLOAT64) throw std::runtime_error("matriz: dtype no soportado");
    const uint64_t esperado = static_cast<uint64_t>(h.filas) * h.columnas * sizeof(double);
    if (h.nbytes != esperado) throw std::runtime_error("matriz: nbytes inconsistente");
    datos.resize(static_cast<size_t>(h.filas) * h.columnas);
    recv_all(s, datos.data(), static_cast<size_t>(h.nbytes));
    return h;
}

} // namespace red