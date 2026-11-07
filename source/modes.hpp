#pragma once

#include <switch.h>
#include <string>

// Run GGML RPC server daemon (worker node on port 50052)
void run_rpc_server_mode(PadState* pad, const std::string& endpoint_host, int port);

// Run lightweight HTTP OpenAI-compatible server on port 8080
void run_http_server_mode(PadState* pad, const std::string& model_path, int port);
