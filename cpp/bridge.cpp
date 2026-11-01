#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include "protocol.hpp"
#include "socket_utils.hpp"

#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

namespace py = pybind11;

namespace {

red::socket_t g_socket = red::SOCKET_INVALIDO;
int g_node_id = -1;

red::socket_t conectar_tcp(const std::string& host, int puerto) {
    red::inicializar_sockets();

    addrinfo hints{};
    hints.ai_family = AF_INET;
    hints.ai_socktype = SOCK_STREAM;

    addrinfo* resultado = nullptr;
    const std::string puerto_txt = std::to_string(puerto);
    const int rc = getaddrinfo(host.c_str(), puerto_txt.c_str(), &hints, &resultado);
    if (rc != 0)
        throw std::runtime_error("getaddrinfo fallo, codigo " + std::to_string(rc));

    red::socket_t s = red::SOCKET_INVALIDO;
    for (addrinfo* p = resultado; p != nullptr; p = p->ai_next) {
        s = ::socket(p->ai_family, p->ai_socktype, p->ai_protocol);
        if (s == red::SOCKET_INVALIDO)
            continue;
        if (::connect(s, p->ai_addr, static_cast<int>(p->ai_addrlen)) == 0)
            break;
        red::cerrar_socket(s);
        s = red::SOCKET_INVALIDO;
    }
    freeaddrinfo(resultado);

    if (s == red::SOCKET_INVALIDO)
        throw std::runtime_error("no se pudo conectar a " + host + ":" + puerto_txt);

    return s;
}

} // namespace

py::array_t<double> eco_matriz(
    py::array_t<double, py::array::c_style | py::array::forcecast> entrada) {

    py::buffer_info buf = entrada.request();
    if (buf.ndim != 2)
        throw std::runtime_error("eco_matriz: se esperaba una matriz 2-D");

    const auto filas = buf.shape[0];
    const auto columnas = buf.shape[1];

    auto salida = py::array_t<double>({filas, columnas});
    py::buffer_info out = salida.request();

    std::memcpy(out.ptr, buf.ptr,static_cast<size_t>(filas * columnas) * sizeof(double));

    return salida;
}

void py_conectar(const std::string& host, int puerto, int node_id, int num_nodos, int timeout_seg) {
    if (g_socket != red::SOCKET_INVALIDO)
        throw std::runtime_error("ya hay una conexion abierta; llama a cerrar()");
    if (num_nodos <= 0 || node_id < 0 || node_id >= num_nodos)
        throw std::runtime_error("node_id/num_nodos invalidos");

    g_socket = conectar_tcp(host, puerto);
    red::configurar_timeout(g_socket, timeout_seg);

    try {
        red::enviar_hello(g_socket, node_id, num_nodos);
    } catch (...) {
        red::cerrar_socket(g_socket);
        g_socket = red::SOCKET_INVALIDO;
        throw;
    }

    g_node_id = node_id;
}

void py_cerrar() {
    if (g_socket != red::SOCKET_INVALIDO) {
        red::cerrar_socket(g_socket);
        g_socket = red::SOCKET_INVALIDO;
    }
    g_node_id = -1;
}

void py_enviar_matriz(
    int ronda, int capa,
    py::array_t<double, py::array::c_style | py::array::forcecast> matriz) {

    if (g_socket == red::SOCKET_INVALIDO)
        throw std::runtime_error("no hay conexion; llama a conectar() primero");
    if (ronda < 0 || capa < 0)
        throw std::runtime_error("ronda/capa deben ser >= 0");

    py::buffer_info buf = matriz.request();
    if (buf.ndim != 2)
        throw std::runtime_error("enviar_matriz: se esperaba una matriz 2-D");

    red::enviar_matriz(g_socket, g_node_id, static_cast<uint32_t>(ronda), static_cast<uint16_t>(capa), static_cast<uint32_t>(buf.shape[0]), static_cast<uint32_t>(buf.shape[1]), static_cast<const double*>(buf.ptr));
}

py::array_t<double> py_recibir_matriz(int ronda, int capa) {
    if (g_socket == red::SOCKET_INVALIDO)
        throw std::runtime_error("no hay conexion; llama a conectar() primero");

    std::vector<double> datos;
    red::MsgHeader h = red::recibir_matriz(g_socket, datos);

    if (h.node_id != static_cast<uint16_t>(g_node_id))
        throw std::runtime_error("recibir_matriz: node_id inesperado");
    if (h.ronda != static_cast<uint32_t>(ronda) ||
        h.capa != static_cast<uint16_t>(capa))
        throw std::runtime_error("recibir_matriz: ronda/capa inesperada");

    auto salida = py::array_t<double>({
        static_cast<py::ssize_t>(h.filas),
        static_cast<py::ssize_t>(h.columnas)
    });
    py::buffer_info out = salida.request();

    std::memcpy(out.ptr, datos.data(), datos.size() * sizeof(double));
    return salida;
}

py::array_t<double> py_sincronizar_matriz(
    int ronda, int capa,
    py::array_t<double, py::array::c_style | py::array::forcecast> matriz) {

    py_enviar_matriz(ronda, capa, matriz);
    return py_recibir_matriz(ronda, capa);
}

PYBIND11_MODULE(nnbridge, m) {
    m.doc() = "Bridge C++ del trabajo final de Redes";

    m.def("eco_matriz", &eco_matriz, "Devuelve una copia de una matriz 2-D (float64)");

    m.def("conectar", &py_conectar, py::arg("host"), py::arg("puerto"), py::arg("node_id"), py::arg("num_nodos"), py::arg("timeout_seg") = 30);

    m.def("enviar_matriz", &py_enviar_matriz, py::arg("ronda"), py::arg("capa"), py::arg("matriz"), "Envia una matriz al servidor central");

    m.def("recibir_matriz", &py_recibir_matriz, py::arg("ronda"), py::arg("capa"), "Recibe la matriz promediada del servidor central");

    m.def("sincronizar_matriz", &py_sincronizar_matriz, py::arg("ronda"), py::arg("capa"), py::arg("matriz"), "Envia la matriz y devuelve la media (conveniencia)");

    m.def("cerrar", &py_cerrar, "Cierra la conexion con el servidor");
}