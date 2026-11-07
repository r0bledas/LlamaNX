#include "network.hpp"
#include <cstdio>
#include <arpa/inet.h>
#include <sys/socket.h>
#include <unistd.h>

static bool s_initialized = false;

bool NetworkManager::init() {
    if (s_initialized) return true;
    
    // Allocate BSD socket buffer (libnx default is 0x200000 = 2MB)
    Result rc = socketInitializeDefault();
    if (R_FAILED(rc)) {
        printf("Failed to initialize sockets: 0x%x\n", rc);
        return false;
    }
    s_initialized = true;
    return true;
}

void NetworkManager::exit() {
    if (s_initialized) {
        socketExit();
        s_initialized = false;
    }
}

bool NetworkManager::isConnected() {
    u32 ip = 0;
    Result rc = nifmGetCurrentIpAddress(&ip);
    return R_SUCCEEDED(rc) && ip != 0;
}

std::string NetworkManager::getLocalIP() {
    u32 ip = 0;
    Result rc = nifmGetCurrentIpAddress(&ip);
    if (R_FAILED(rc) || ip == 0) {
        return "Unknown / Disconnected";
    }
    struct in_addr addr;
    addr.s_addr = ip;
    return std::string(inet_ntoa(addr));
}
