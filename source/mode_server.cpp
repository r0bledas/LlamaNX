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

void run_http_server_mode(PadState* pad, const std::string& model_path, int port) {
    consoleClear();
    printf("\x1b[1;36m=== LlamaNX - HTTP OpenAI Server ===\x1b[0m\n\n");

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
    printf("Model: \x1b[1;32m%s\x1b[0m\n", model_path.c_str());
    printf("Endpoint: \x1b[1;32mhttp://%s:%d\x1b[0m\n\n", ip.c_str(), port);
    printf("Test with curl:\n");
    printf("  \x1b[1;37mcurl http://%s:%d/health\x1b[0m\n\n", ip.c_str(), port);
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

    int flags = fcntl(server_fd, F_GETFL, 0);
    fcntl(server_fd, F_SETFL, flags | O_NONBLOCK);

    struct sockaddr_in address;
    memset(&address, 0, sizeof(address));
    address.sin_family = AF_INET;
    address.sin_addr.s_addr = INADDR_ANY;
    address.sin_port = htons(port);

    if (bind(server_fd, (struct sockaddr*)&address, sizeof(address)) < 0) {
        printf("[ERROR] Bind failed on port %d: %d\n", port, errno);
        close(server_fd);
        consoleUpdate(NULL);
        return;
    }

    if (listen(server_fd, 8) < 0) {
        printf("[ERROR] Listen failed: %d\n", errno);
        close(server_fd);
        consoleUpdate(NULL);
        return;
    }

    printf("[HTTP Server] Ready and listening for requests...\n");
    consoleUpdate(NULL);

    while (appletMainLoop()) {
        padUpdate(pad);
        u64 kDown = padGetButtonsDown(pad);
        if (kDown & HidNpadButton_Plus) {
            printf("\n[HTTP Server] Stopping server...\n");
            consoleUpdate(NULL);
            break;
        }

        struct sockaddr_in client_addr;
        socklen_t addrlen = sizeof(client_addr);
        int client_fd = accept(server_fd, (struct sockaddr*)&client_addr, &addrlen);

        if (client_fd >= 0) {
            char buffer[1024];
            int bytes_read = recv(client_fd, buffer, sizeof(buffer) - 1, 0);
            if (bytes_read > 0) {
                buffer[bytes_read] = '\0';
                
                // Simple HTTP responder for /health & discovery
                const char* body = "{\"status\":\"ok\",\"model\":\"llamanx\",\"engine\":\"llama.cpp-switch\"}";
                char response[512];
                snprintf(response, sizeof(response),
                    "HTTP/1.1 200 OK\r\n"
                    "Content-Type: application/json\r\n"
                    "Content-Length: %zu\r\n"
                    "Access-Control-Allow-Origin: *\r\n"
                    "Connection: close\r\n\r\n%s",
                    strlen(body), body);
                
                send(client_fd, response, strlen(response), 0);
                printf("[HTTP] 200 OK responded to incoming client\n");
            }
            close(client_fd);
        }

        svcSleepThread(10000000ULL); // 10ms
        consoleUpdate(NULL);
    }

    close(server_fd);
    printf("[HTTP Server] Closed.\n");
    consoleUpdate(NULL);
}
