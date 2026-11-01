#pragma once

#include <cerrno>
#include <cstring>
#include <stdexcept>
#include <string>

#ifdef _WIN32
    #include <winsock2.h>
    #include <ws2tcpip.h>
    #pragma comment(lib, "ws2_32.lib")
#else
    #include <arpa/inet.h>
    #include <netdb.h>
    #include <netinet/in.h>
    #include <netinet/tcp.h>
    #include <signal.h>
    #include <sys/socket.h>
    #include <sys/time.h>
    #include <sys/types.h>
    #include <unistd.h>
#endif

namespace red {

#ifdef _WIN32
    using socket_t = SOCKET;
    static constexpr socket_t SOCKET_INVALIDO = INVALID_SOCKET;
#else
    using socket_t = int;
    static constexpr socket_t SOCKET_INVALIDO = -1;
#endif

#ifdef _WIN32
struct InicializarWinsock {
    InicializarWinsock() {
        WSADATA datos;
        if (WSAStartup(MAKEWORD(2, 2), &datos) != 0)
            throw std::runtime_error("WSAStartup fallo");
    }
    ~InicializarWinsock() { WSACleanup(); }
};
#endif

inline void inicializar_sockets() {
#ifdef _WIN32
    static InicializarWinsock init;
    (void)init;
#else
    signal(SIGPIPE, SIG_IGN);
#endif
}

inline void cerrar_socket(socket_t s) {
    if (s == SOCKET_INVALIDO) return;
#ifdef _WIN32
    closesocket(s);
#else
    ::close(s);
#endif
}

inline std::string error_socket() {
#ifdef _WIN32
    return "WSA error " + std::to_string(WSAGetLastError());
#else
    return std::strerror(errno);
#endif
}

inline void configurar_timeout(socket_t s, int segundos) {
#ifdef _WIN32
    DWORD ms = static_cast<DWORD>(segundos) * 1000;
    setsockopt(s, SOL_SOCKET, SO_RCVTIMEO, reinterpret_cast<const char*>(&ms), sizeof(ms));
    setsockopt(s, SOL_SOCKET, SO_SNDTIMEO, reinterpret_cast<const char*>(&ms), sizeof(ms));
#else
    timeval tv{};
    tv.tv_sec = segundos;
    tv.tv_usec = 0;
    setsockopt(s, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));
    setsockopt(s, SOL_SOCKET, SO_SNDTIMEO, &tv, sizeof(tv));
#endif
}

} // namespace red