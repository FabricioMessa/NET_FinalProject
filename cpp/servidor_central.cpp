#include "protocol.hpp"
#include "socket_utils.hpp"

#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace {
    red::socket_t crear_listen_socket(int puerto) {
        red::inicializar_sockets();

        red::socket_t s = ::socket(AF_INET, SOCK_STREAM, 0);
        if (s == red::SOCKET_INVALIDO)
            throw std::runtime_error("socket: " + red::error_socket());

        int uno = 1;
        setsockopt(s, SOL_SOCKET, SO_REUSEADDR, reinterpret_cast<const char*>(&uno), sizeof(uno));

        sockaddr_in addr{};
        addr.sin_family = AF_INET;
        addr.sin_addr.s_addr = htonl(INADDR_ANY);
        addr.sin_port = htons(static_cast<uint16_t>(puerto));

        if (::bind(s, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) < 0)
            throw std::runtime_error("bind: " + red::error_socket());

        if (::listen(s, 16) < 0)
            throw std::runtime_error("listen: " + red::error_socket());

        return s;
    }

    red::socket_t aceptar_con_timeout(red::socket_t listen_sock, int segundos) {
        red::configurar_timeout(listen_sock, segundos);
        sockaddr_in cliente{};
    #ifdef _WIN32
        int len = sizeof(cliente);
    #else
        socklen_t len = sizeof(cliente);
    #endif
        return ::accept(listen_sock, reinterpret_cast<sockaddr*>(&cliente), &len);
    }

    } // namespace

    int main(int argc, char** argv) {
        try {
            const int puerto    = (argc > 1) ? std::atoi(argv[1]) : 5000;
            const int num_nodos = (argc > 2) ? std::atoi(argv[2]) : 3;
            const int rondas    = (argc > 3) ? std::atoi(argv[3]) : 10;
            const int num_capas = (argc > 4) ? std::atoi(argv[4]) : 3;

            if (num_nodos <= 0 || rondas <= 0 || num_capas <= 0)
                throw std::runtime_error("parametros invalidos");

            red::socket_t listen_sock = crear_listen_socket(puerto);
            std::cout << "[servidor] puerto=" << puerto << " nodos=" << num_nodos << " rondas=" << rondas << " capas=" << num_capas << std::endl;

            std::vector<red::socket_t> nodos(num_nodos, red::SOCKET_INVALIDO);
            int conectados = 0;
            while (conectados < num_nodos) {
                red::socket_t s = aceptar_con_timeout(listen_sock, 30);
                if (s == red::SOCKET_INVALIDO)
                    throw std::runtime_error("timeout esperando nodos");

                red::configurar_timeout(s, 30);
                red::Hello h = red::recibir_hello(s);
                if (h.num_nodos != num_nodos || h.node_id >= num_nodos) {
                    std::cerr << "[servidor] hello invalido" << std::endl;
                    red::cerrar_socket(s);
                    continue;
                }
                if (nodos[h.node_id] != red::SOCKET_INVALIDO) {
                    std::cerr << "[servidor] node_id duplicado " << h.node_id << std::endl;
                    red::cerrar_socket(s);
                    continue;
                }
                nodos[h.node_id] = s;
                ++conectados;
                std::cout << "[servidor] nodo " << h.node_id << " conectado (" << conectados << "/" << num_nodos << ")" << std::endl;
            }

            for (auto& s : nodos) red::configurar_timeout(s, 60);

            for (int ronda = 0; ronda < rondas; ++ronda) {
                for (int capa = 0; capa < num_capas; ++capa) {
                    std::vector<std::vector<double>> matrices(num_nodos);
                    uint32_t filas = 0, columnas = 0;

                    for (int i = 0; i < num_nodos; ++i) {
                        red::MsgHeader h = red::recibir_matriz(nodos[i], matrices[i]);
                        if (h.node_id != i)
                            throw std::runtime_error("orden de nodos inesperado");
                        if (h.ronda != static_cast<uint32_t>(ronda) ||
                            h.capa != static_cast<uint16_t>(capa))
                            throw std::runtime_error("ronda/capa inesperada");
                        if (i == 0) {
                            filas = h.filas;
                            columnas = h.columnas;
                        } else if (h.filas != filas || h.columnas != columnas) {
                            throw std::runtime_error("shapes distintas entre nodos");
                        }
                    }

                    std::vector<double> media(static_cast<size_t>(filas) * columnas, 0.0);
                    for (int i = 0; i < num_nodos; ++i)
                        for (size_t k = 0; k < media.size(); ++k)
                            media[k] += matrices[i][k];
                    for (double& v : media) v /= num_nodos;

                    for (int i = 0; i < num_nodos; ++i)
                        red::enviar_matriz(nodos[i], i, ronda, static_cast<uint16_t>(capa), filas, columnas, media.data());
                }
                std::cout << "[servidor] ronda " << (ronda + 1) << "/" << rondas << " OK" << std::endl;
            }

            for (auto s : nodos) red::cerrar_socket(s);
            red::cerrar_socket(listen_sock);
            std::cout << "[servidor] listo" << std::endl;
            return 0;
        } catch (const std::exception& e) {
            std::cerr << "[servidor] ERROR: " << e.what() << std::endl;
            return 1;
        }
}