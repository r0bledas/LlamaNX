#include "network.hpp"
#include <cstdio>
#include <arpa/inet.h>
#include <sys/socket.h>
#include <unistd.h>

static bool s_socket_initialized = false;
static bool s_nifm_initialized = false;

bool NetworkManager::init() {
    // 1. Initialize NIFM service session so nifmGetCurrentIpAddress works
    if (!s_nifm_initialized) {
        Result nifm_rc = nifmInitialize(NifmServiceType_User);
        if (R_SUCCEEDED(nifm_rc)) {
            s_nifm_initialized = true;
        } else {
            printf("[Network] nifmInitialize failed: 0x%x\n", nifm_rc);
        }
    }

    // 2. Initialize BSD socket service
    if (!s_socket_initialized) {
        Result sock_rc = socketInitializeDefault();
        if (R_FAILED(sock_rc)) {
            printf("[Network] socketInitializeDefault failed: 0x%x\n", sock_rc);
            return false;
        }
        s_socket_initialized = true;
    }

    return true;
}

void NetworkManager::exit() {
    if (s_socket_initialized) {
        socketExit();
        s_socket_initialized = false;
    }
    if (s_nifm_initialized) {
        nifmExit();
        s_nifm_initialized = false;
    }
}

bool NetworkManager::isConnected() {
    u32 ip = 0;
    Result rc = nifmGetCurrentIpAddress(&ip);
    return R_SUCCEEDED(rc) && ip != 0;
}

std::string NetworkManager::getLocalIP() {
    if (!s_nifm_initialized) {
        Result nifm_rc = nifmInitialize(NifmServiceType_User);
        if (R_SUCCEEDED(nifm_rc)) {
            s_nifm_initialized = true;
        }
    }

    u32 ip = 0;
    Result rc = nifmGetCurrentIpAddress(&ip);
    if (R_SUCCEEDED(rc) && ip != 0) {
        struct in_addr addr;
        addr.s_addr = ip;
        return std::string(inet_ntoa(addr));
    }

    // Fallback: use gethostid() which also queries the local IP in libnx
    unsigned long host_id = gethostid();
    if (host_id != 0 && host_id != 0x7f000001) {
        struct in_addr addr;
        addr.s_addr = (u32)host_id;
        return std::string(inet_ntoa(addr));
    }

    return "No IP (Check Wi-Fi)";
}
