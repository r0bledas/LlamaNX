#include "modes.hpp"
#include "network.hpp"
#include "web_assets.hpp"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <sstream>
#include <dirent.h>
#include <sys/stat.h>
#include <unistd.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <fcntl.h>
#include <errno.h>

#include <ggml.h>
#include <gguf.h>
#include <llama.h>

namespace {
    struct ServerConfig {
        std::string loaded_model_path;
        int n_ctx = 1024;
        int n_threads = 3;
        int n_predict = 256;
        float temp = 0.7f;
    } s_cfg;

    llama_model* s_model = nullptr;
    llama_context* s_ctx = nullptr;
    const llama_vocab* s_vocab = nullptr;

    struct ModelItem {
        std::string name;
        std::string path;
        u64 size;
    };

    std::vector<ModelItem> scan_models(const char* dir) {
        std::vector<ModelItem> out;
        DIR* d = opendir(dir);
        if (!d) return out;
        struct dirent* e;
        while ((e = readdir(d)) != nullptr) {
            std::string name = e->d_name;
            if (name.size() > 5 && name.compare(name.size() - 5, 5, ".gguf") == 0) {
                std::string path = std::string(dir) + "/" + name;
                struct stat st;
                u64 sz = (stat(path.c_str(), &st) == 0) ? (u64)st.st_size : 0;
                out.push_back({ name, path, sz });
            }
        }
        closedir(d);
        return out;
    }

    std::vector<ModelItem> get_all_models() {
        std::vector<ModelItem> m = scan_models("sdmc:/switch/llamanx");
        std::vector<ModelItem> m2 = scan_models("sdmc:/switch/chatnx");
        m.insert(m.end(), m2.begin(), m2.end());
        return m;
    }

    void free_current_model() {
        if (s_ctx) {
            llama_free(s_ctx);
            s_ctx = nullptr;
        }
        if (s_model) {
            llama_model_free(s_model);
            s_model = nullptr;
        }
    }

    bool load_model_instance(const std::string& path, int n_ctx, int n_threads) {
        free_current_model();
        if (path.empty()) return false;

        llama_backend_init();
        llama_model_params mp = llama_model_default_params();
        mp.load_mode = LLAMA_LOAD_MODE_NONE; // standard buffered read (safe for FAT32 / libnx)

        printf("[Web Server] Loading %s (ctx: %d, th: %d)...\n", path.c_str(), n_ctx, n_threads);
        consoleUpdate(NULL);

        s_model = llama_model_load_from_file(path.c_str(), mp);
        if (!s_model) {
            printf("[Web Server] Failed to load model\n");
            consoleUpdate(NULL);
            return false;
        }

        llama_context_params cp = llama_context_default_params();
        cp.n_ctx = n_ctx;
        cp.n_threads = n_threads;
        cp.n_threads_batch = n_threads;

        s_ctx = llama_init_from_model(s_model, cp);
        if (!s_ctx) {
            llama_model_free(s_model);
            s_model = nullptr;
            printf("[Web Server] Failed to create context\n");
            consoleUpdate(NULL);
            return false;
        }

        s_vocab = llama_model_get_vocab(s_model);
        s_cfg.loaded_model_path = path;
        s_cfg.n_ctx = n_ctx;
        s_cfg.n_threads = n_threads;
        printf("[Web Server] Model loaded successfully!\n");
        consoleUpdate(NULL);
        return true;
    }

    std::string run_inference(const std::string& user_prompt, int max_tokens, float temp) {
        if (!s_ctx || !s_vocab) return "No model loaded. Please select and load a model from the web sidebar.";

        llama_sampler* smpl = llama_sampler_chain_init(llama_sampler_chain_default_params());
        llama_sampler_chain_add(smpl, llama_sampler_init_temp(temp));
        llama_sampler_chain_add(smpl, llama_sampler_init_dist(42));

        const char* tmpl = llama_model_chat_template(s_model, nullptr);
        std::vector<llama_chat_message> messages = {
            { "user", user_prompt.c_str() }
        };

        std::vector<char> formatted(4096);
        int new_len = llama_chat_apply_template(tmpl, messages.data(), messages.size(),
                                                true, formatted.data(), formatted.size());
        if (new_len > (int)formatted.size()) {
            formatted.resize(new_len);
            new_len = llama_chat_apply_template(tmpl, messages.data(), messages.size(),
                                                true, formatted.data(), formatted.size());
        }

        std::string prompt_str = (new_len > 0) ? std::string(formatted.data(), new_len) : user_prompt;

        int n_tokens = -llama_tokenize(s_vocab, prompt_str.c_str(), (int32_t)prompt_str.size(), nullptr, 0, true, true);
        std::vector<llama_token> tokens(n_tokens);
        int got = llama_tokenize(s_vocab, prompt_str.c_str(), (int32_t)prompt_str.size(), tokens.data(), n_tokens, true, true);
        tokens.resize(got < 0 ? 0 : got);

        llama_batch batch = llama_batch_get_one(tokens.data(), tokens.size());
        if (llama_decode(s_ctx, batch) != 0) {
            llama_sampler_free(smpl);
            return "[Error during initial decode]";
        }

        std::string response;
        for (int i = 0; i < max_tokens; ++i) {
            llama_token tok = llama_sampler_sample(smpl, s_ctx, -1);
            if (llama_vocab_is_eog(s_vocab, tok)) break;

            char buf[256];
            int n = llama_token_to_piece(s_vocab, tok, buf, sizeof(buf), 0, true);
            if (n > 0) response.append(buf, n);

            batch = llama_batch_get_one(&tok, 1);
            if (llama_decode(s_ctx, batch) != 0) break;
        }

        llama_sampler_free(smpl);
        return response;
    }

    std::string escape_json(const std::string& s) {
        std::ostringstream o;
        for (char c : s) {
            if (c == '"') o << "\\\"";
            else if (c == '\\') o << "\\\\";
            else if (c == '\b') o << "\\b";
            else if (c == '\f') o << "\\f";
            else if (c == '\n') o << "\\n";
            else if (c == '\r') o << "\\r";
            else if (c == '\t') o << "\\t";
            else if ((unsigned char)c < ' ') o << "\\u" << std::hex << (int)c;
            else o << c;
        }
        return o.str();
    }
}

void run_http_server_mode(PadState* pad, const std::string& initial_model, int port) {
    consoleClear();
    printf("\x1b[1;36m====================================================\x1b[0m\n");
    printf("\x1b[1;37m        LlamaNX Web Remote Control & Server         \x1b[0m\n");
    printf("\x1b[1;36m====================================================\x1b[0m\n\n");

    if (!NetworkManager::init()) {
        printf("\x1b[1;31m[ERROR] Failed to start network stack.\x1b[0m\n");
        printf("Press (+) to return.\n");
        consoleUpdate(NULL);
        while (appletMainLoop()) {
            padUpdate(pad);
            if (padGetButtonsDown(pad) & HidNpadButton_Plus) break;
        }
        return;
    }

    std::string ip = NetworkManager::getLocalIP();
    printf("Access Web UI on your phone or PC:\n");
    printf("  \x1b[1;32mhttp://%s:%d\x1b[0m\n\n", ip.c_str(), port);
    printf("Hardware: 4x ARM Cortex-A57 (Tegra X1)\n");
    printf("Controls: Press \x1b[1;31m(+)\x1b[0m to stop server.\n");
    printf("----------------------------------------------------\n");
    consoleUpdate(NULL);

    if (!initial_model.empty()) {
        load_model_instance(initial_model, s_cfg.n_ctx, s_cfg.n_threads);
    }

    int server_fd = socket(AF_INET, SOCK_STREAM, 0);
    if (server_fd < 0) {
        printf("[ERROR] Failed to create socket\n");
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
        printf("[ERROR] Bind failed on port %d\n", port);
        close(server_fd);
        consoleUpdate(NULL);
        return;
    }

    listen(server_fd, 8);
    printf("[Ready] Waiting for incoming web connections...\n");
    consoleUpdate(NULL);

    while (appletMainLoop()) {
        padUpdate(pad);
        u64 kDown = padGetButtonsDown(pad);
        if (kDown & HidNpadButton_Plus) {
            printf("\nStopping server...\n");
            consoleUpdate(NULL);
            break;
        }

        struct sockaddr_in client_addr;
        socklen_t addrlen = sizeof(client_addr);
        int client_fd = accept(server_fd, (struct sockaddr*)&client_addr, &addrlen);

        if (client_fd >= 0) {
            std::vector<char> req_buf(16384);
            int bytes_read = recv(client_fd, req_buf.data(), req_buf.size() - 1, 0);
            if (bytes_read > 0) {
                req_buf[bytes_read] = '\0';
                std::string req(req_buf.data());

                std::string res_header;
                std::string res_body;

                if (req.rfind("GET / ", 0) == 0 || req.rfind("GET /index.html", 0) == 0) {
                    // Serve embedded lightweight HTML5 Web Remote Control
                    res_body = INDEX_HTML;
                    res_header = "HTTP/1.1 200 OK\r\nContent-Type: text/html\r\nConnection: close\r\n\r\n";
                } else if (req.rfind("GET /api/status", 0) == 0) {
                    u64 total_ram = 0, used_ram = 0;
                    svcGetInfo(&total_ram, InfoType_TotalMemorySize, CUR_PROCESS_HANDLE, 0);
                    svcGetInfo(&used_ram,  InfoType_UsedMemorySize,  CUR_PROCESS_HANDLE, 0);

                    std::ostringstream json;
                    json << "{"
                         << "\"ip\":\"" << ip << "\","
                         << "\"total_ram_mb\":" << (total_ram / (1024 * 1024)) << ","
                         << "\"used_ram_mb\":" << (used_ram / (1024 * 1024)) << ","
                         << "\"loaded_model\":\"" << escape_json(s_cfg.loaded_model_path) << "\","
                         << "\"models\":[";
                    
                    std::vector<ModelItem> models = get_all_models();
                    for (size_t i = 0; i < models.size(); ++i) {
                        json << "{\"name\":\"" << escape_json(models[i].name) << "\","
                             << "\"path\":\"" << escape_json(models[i].path) << "\","
                             << "\"size\":" << models[i].size << "}";
                        if (i + 1 < models.size()) json << ",";
                    }
                    json << "]}";

                    res_body = json.str();
                    res_header = "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nAccess-Control-Allow-Origin: *\r\nConnection: close\r\n\r\n";
                } else if (req.rfind("POST /api/config", 0) == 0) {
                    // Simple parse of model path, context, threads
                    size_t body_pos = req.find("\r\n\r\n");
                    std::string body = (body_pos != std::string::npos) ? req.substr(body_pos + 4) : "";

                    std::string target_model;
                    int n_ctx = s_cfg.n_ctx;
                    int th = s_cfg.n_threads;

                    size_t m_pos = body.find("\"model\":\"");
                    if (m_pos != std::string::npos) {
                        size_t m_end = body.find("\"", m_pos + 9);
                        if (m_end != std::string::npos) {
                            target_model = body.substr(m_pos + 9, m_end - (m_pos + 9));
                        }
                    }
                    size_t c_pos = body.find("\"n_ctx\":");
                    if (c_pos != std::string::npos) n_ctx = atoi(body.c_str() + c_pos + 8);
                    size_t t_pos = body.find("\"threads\":");
                    if (t_pos != std::string::npos) th = atoi(body.c_str() + t_pos + 10);

                    if (n_ctx <= 0) n_ctx = 1024;
                    if (th <= 0 || th > 4) th = 3;

                    bool ok = load_model_instance(target_model, n_ctx, th);
                    res_body = ok ? "{\"success\":true,\"message\":\"Model loaded successfully\"}"
                                  : "{\"success\":false,\"message\":\"Failed to load model\"}";
                    res_header = "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nAccess-Control-Allow-Origin: *\r\nConnection: close\r\n\r\n";
                } else if (req.rfind("POST /v1/chat/completions", 0) == 0) {
                    size_t body_pos = req.find("\r\n\r\n");
                    std::string body = (body_pos != std::string::npos) ? req.substr(body_pos + 4) : "";

                    // Extract user message content
                    std::string prompt = "Hello";
                    size_t c_pos = body.find("\"content\":");
                    if (c_pos != std::string::npos) {
                        size_t start = body.find("\"", c_pos + 10);
                        if (start != std::string::npos) {
                            size_t end = body.find("\"", start + 1);
                            if (end != std::string::npos) prompt = body.substr(start + 1, end - start - 1);
                        }
                    }

                    printf("[Web] Generating response for: %s\n", prompt.c_str());
                    consoleUpdate(NULL);

                    std::string answer = run_inference(prompt, s_cfg.n_predict, s_cfg.temp);

                    std::ostringstream json;
                    json << "{"
                         << "\"id\":\"chatcmpl-llamanx\","
                         << "\"object\":\"chat.completion\","
                         << "\"choices\":[{"
                         << "\"message\":{\"role\":\"assistant\",\"content\":\"" << escape_json(answer) << "\"},"
                         << "\"finish_reason\":\"stop\""
                         << "}]}";

                    res_body = json.str();
                    res_header = "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nAccess-Control-Allow-Origin: *\r\nConnection: close\r\n\r\n";
                } else {
                    res_body = "{\"status\":\"ok\",\"service\":\"LlamaNX\"}";
                    res_header = "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nConnection: close\r\n\r\n";
                }

                send(client_fd, res_header.c_str(), res_header.size(), 0);
                send(client_fd, res_body.c_str(), res_body.size(), 0);
            }
            close(client_fd);
        }

        svcSleepThread(10000000ULL); // 10ms
        consoleUpdate(NULL);
    }

    free_current_model();
    close(server_fd);
    printf("[Web Server] Stopped.\n");
    consoleUpdate(NULL);
}
