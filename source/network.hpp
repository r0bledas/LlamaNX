#pragma once

#include <string>
#include <switch.h>

class NetworkManager {
public:
    static bool init();
    static void exit();
    static bool isConnected();
    static std::string getLocalIP();
};
