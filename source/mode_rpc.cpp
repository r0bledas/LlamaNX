#include "modes.hpp"
#include "network.hpp"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <unistd.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <fcntl.h>
#include <errno.h>

void run_rpc_server_mode(PadState* pad, const std::string& host_ip, int port) {
    consoleClear();
    printf("\x1b[1;36m=== LlamaNX - GGML RPC Server Node ===\x1b[0m\n\n");

    if (!NetworkManager::init()) {
        printf("\x1b[1;31m[ERROR] Could not initialize network stack.\x1b[0m\n");
        printf("Press + to return.\n");
        consoleUpdate(NULL);
        while (appletMainLoop()) {
            padUpdate(pad);
            if (padGetButtonsDown(pad) & HidNpadButton_Plus) break;
        }
        return;
    }

    std::string ip = NetworkManager::getLocalIP();
    printf("Switch IP: \x1b[1;32m%s\x1b[0m\n", ip.c_str());
    printf("Listening on Port: \x1b[1;33m%d\x1b[0m\n\n", port);
    printf("On your host PC, connect using llama.cpp via:\n");
    printf("  \x1b[1;37mllama-cli -m model.gguf --rpc %s:%d -ngl 4\x1b[0m\n\n", ip.c_str(), port);
    printf("Controls: Press \x1b[1;31m(+)\x1b[0m to stop server.\n");
    printf("--------------------------------------------------\n");
    consoleUpdate(NULL);

    int server_fd = socket(AF_INET, SOCK_STREAM, 0);
    if (server_fd < 0) {
        printf("[ERROR] Failed to create socket: %d\n", errno);
        consoleUpdate(NULL);
        return;
    }

    int opt = 1;
    setsockopt(server_fd, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));

    // Set non-blocking so the main applet loop can poll buttons
    int flags = fcntl(server_fd, F_GETFL, 0);
    fcntl(server_fd, F_SETFL, flags | O_NONBLOCK);

    struct sockaddr_in address;
    memset(&address, 0, sizeof(address));
    address.sin_family = AF_INET;
    address.sin_addr.s_addr = INADDR_ANY;
    address.sin_port = htons(port);

    if (bind(server_fd, (struct sockaddr*)&address, sizeof(address)) < 0) {
        printf("[ERROR] Socket bind failed on port %d: %d\n", port, errno);
        close(server_fd);
        consoleUpdate(NULL);
        return;
    }

    if (listen(server_fd, 4) < 0) {
        printf("[ERROR] Socket listen failed: %d\n", errno);
        close(server_fd);
        consoleUpdate(NULL);
        return;
    }

    printf("[RPC Node] Waiting for connections from host PC...\n");
    consoleUpdate(NULL);

    while (appletMainLoop()) {
        padUpdate(pad);
        u64 kDown = padGetButtonsDown(pad);
        if (kDown & HidNpadButton_Plus) {
            printf("\n[RPC Node] Stopping server requested by user...\n");
            consoleUpdate(NULL);
            break;
        }

        struct sockaddr_in client_addr;
        socklen_t addrlen = sizeof(client_addr);
        int client_fd = accept(server_fd, (struct sockaddr*)&client_addr, &addrlen);

        if (client_fd >= 0) {
            char client_ip[INET_ADDRSTRLEN];
            inet_ntop(AF_INET, &client_addr.sin_addr, client_ip, INET_ADDRSTRLEN);
            printf("[RPC Node] \x1b[1;32mClient connected from %s\x1b[0m\n", client_ip);
            consoleUpdate(NULL);

            // In actual GGML RPC backend handling, ggml_backend_rpc_serve handles the socket loop
            close(client_fd);
        } else if (errno != EAGAIN && errno != EWOULDBLOCK) {
            printf("[RPC Node] Accept warning: errno %d\n", errno);
            consoleUpdate(NULL);
        }

        svcSleepThread(10000000ULL); // 10ms
        consoleUpdate(NULL);
    }

    close(server_fd);
    printf("[RPC Node] Server closed.\n");
    consoleUpdate(NULL);
}
