#include "modes.hpp"
#include "network.hpp"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include <dirent.h>
#include <sys/stat.h>

#include <switch.h>

#include <ggml.h>
#include <gguf.h>
#include <llama.h>

extern "C" {
    extern char* fake_heap_start;
    extern char* fake_heap_end;
}

namespace {
    constexpr const char* MODEL_DIR_1 = "sdmc:/switch/llamanx";
    constexpr const char* MODEL_DIR_2 = "sdmc:/switch/chatnx";

    constexpr int N_CTX = 2048;
    constexpr int N_THREADS = 3;
    constexpr int N_PREDICT = 512;

    struct ModelEntry {
        std::string name;
        std::string path;
        std::string quant;
        u64         size;
    };

    std::string detect_quant(const std::string& path) {
        struct gguf_init_params gp = { true, nullptr };
        gguf_context* g = gguf_init_from_file(path.c_str(), gp);
        if (!g) return "?";

        int counts[64] = {0};
        int64_t n = gguf_get_n_tensors(g);
        for (int64_t i = 0; i < n; ++i) {
            int t = (int)gguf_get_tensor_type(g, i);
            if (t >= 0 && t < 64) counts[t]++;
        }
        gguf_free(g);

        int best = -1, best_n = 0, any = -1, any_n = 0;
        for (int t = 0; t < 64; ++t) {
            if (counts[t] > any_n) { any_n = counts[t]; any = t; }
            if (t == (int)GGML_TYPE_F32 || t == (int)GGML_TYPE_F16) continue;
            if (counts[t] > best_n) { best_n = counts[t]; best = t; }
        }
        int chosen = best >= 0 ? best : any;
        if (chosen < 0) return "?";
        return ggml_type_name((enum ggml_type)chosen);
    }

    u64 file_size(const std::string& path) {
        struct stat st;
        return stat(path.c_str(), &st) == 0 ? (u64)st.st_size : 0;
    }

    std::vector<ModelEntry> list_models(const char* dir) {
        std::vector<ModelEntry> out;
        DIR* d = opendir(dir);
        if (!d) return out;
        struct dirent* e;
        while ((e = readdir(d)) != nullptr) {
            std::string name = e->d_name;
            if (name.size() > 5 && name.compare(name.size() - 5, 5, ".gguf") == 0) {
                std::string path = std::string(dir) + "/" + name;
                out.push_back({ name, path, detect_quant(path), file_size(path) });
            }
        }
        closedir(d);
        return out;
    }

    std::vector<ModelEntry> scan_all_models() {
        std::vector<ModelEntry> models = list_models(MODEL_DIR_1);
        std::vector<ModelEntry> legacy = list_models(MODEL_DIR_2);
        models.insert(models.end(), legacy.begin(), legacy.end());
        return models;
    }

    bool is_application_mode() {
        AppletType t = appletGetAppletType();
        return t == AppletType_Application || t == AppletType_SystemApplication;
    }

    // Accurate heap inspection from Newlib fake_heap boundaries
    u64 get_heap_total_bytes() {
        if (fake_heap_start && fake_heap_end && fake_heap_end > fake_heap_start) {
            return (u64)(fake_heap_end - fake_heap_start);
        }
        // Fallback to svcGetInfo
        u64 total = 0;
        svcGetInfo(&total, InfoType_TotalMemorySize, CUR_PROCESS_HANDLE, 0);
        return total;
    }

    void log_cb(ggml_log_level level, const char* text, void*) {
        // Print errors and warnings to Switch screen console
        if (level == GGML_LOG_LEVEL_ERROR || level == GGML_LOG_LEVEL_WARN) {
            printf("%s", text);
            consoleUpdate(NULL);
        }
    }

    void wait_for_exit(PadState* pad) {
        printf("\nPress (+) to exit.\n");
        consoleUpdate(NULL);
        while (appletMainLoop()) {
            padUpdate(pad);
            if (padGetButtonsDown(pad) & HidNpadButton_Plus) break;
            consoleUpdate(NULL);
        }
    }

    std::vector<llama_token> tokenize(const llama_vocab* vocab, const std::string& text,
                                      bool add_special) {
        int n = -llama_tokenize(vocab, text.c_str(), (int32_t)text.size(),
                                nullptr, 0, add_special, true);
        std::vector<llama_token> out(n);
        int got = llama_tokenize(vocab, text.c_str(), (int32_t)text.size(),
                                 out.data(), n, add_special, true);
        out.resize(got < 0 ? 0 : got);
        return out;
    }

    std::string token_to_text(const llama_vocab* vocab, llama_token tok) {
        char buf[256];
        int n = llama_token_to_piece(vocab, tok, buf, sizeof(buf), 0, true);
        if (n < 0) return std::string();
        return std::string(buf, n);
    }

    bool read_line(const char* guide, std::string& out) {
        SwkbdConfig kbd;
        if (R_FAILED(swkbdCreate(&kbd, 0))) return false;
        swkbdConfigMakePresetDefault(&kbd);
        swkbdConfigSetHeaderText(&kbd, "LlamaNX Prompt");
        swkbdConfigSetGuideText(&kbd, guide);

        char buf[1024] = {0};
        Result rc = swkbdShow(&kbd, buf, sizeof(buf));
        swkbdClose(&kbd);
        if (R_FAILED(rc)) return false;

        out.assign(buf);
        return true;
    }

    struct ChatView {
        struct Line { std::string s; int len; };
        std::vector<Line> lines;
        int scroll = 0;
        int max_visible = 38;

        void clear() { lines.clear(); scroll = 0; }

        void ensure_line() {
            if (lines.empty()) {
                lines.push_back({ "", 0 });
            }
        }

        void append(const std::string& text) {
            ensure_line();
            for (size_t i = 0; i < text.size(); ++i) {
                if (text[i] == '\x1b' && i + 1 < text.size() && text[i+1] == '[') {
                    size_t j = i + 2;
                    while (j < text.size() && text[j] != 'm') ++j;
                    if (j < text.size()) {
                        lines.back().s.append(text, i, j - i + 1);
                        i = j;
                        continue;
                    }
                }
                if (text[i] == '\n') {
                    lines.push_back({ "", 0 });
                } else {
                    if (lines.back().len >= 78) {
                        lines.push_back({ "", 0 });
                    }
                    lines.back().s.push_back(text[i]);
                    lines.back().len++;
                }
            }
            to_bottom();
        }

        void to_bottom() {
            int n = (int)lines.size();
            scroll = n > max_visible ? n - max_visible : 0;
        }

        void render() {
            consoleClear();
            int n = (int)lines.size();
            int end = scroll + max_visible;
            if (end > n) end = n;
            for (int i = scroll; i < end; ++i) {
                printf("%s\n", lines[i].s.c_str());
            }
            consoleUpdate(NULL);
        }
    } g_view;

    int select_model(PadState* pad, const std::vector<ModelEntry>& models, std::string& out) {
        int selected = 0;
        int n = (int)models.size();
        while (appletMainLoop()) {
            consoleClear();
            printf("\x1b[1;36m=== LlamaNX - Select GGUF Model ===\x1b[0m\n\n");
            for (int i = 0; i < n; ++i) {
                double mb = (double)models[i].size / (1024.0 * 1024.0);
                if (i == selected) {
                    printf("\x1b[7m > %s (%s, %.1f MB) \x1b[0m\n",
                           models[i].name.c_str(), models[i].quant.c_str(), mb);
                } else {
                    printf("   %s (%s, %.1f MB)\n",
                           models[i].name.c_str(), models[i].quant.c_str(), mb);
                }
            }
            printf("\n\x1b[2;37m(Up/Down): Select | (A): Confirm | (+): Exit\x1b[0m\n");
            consoleUpdate(NULL);

            padUpdate(pad);
            u64 kDown = padGetButtonsDown(pad);
            if (kDown & HidNpadButton_Plus) return -1;
            if ((kDown & HidNpadButton_Up)   && selected > 0)     selected--;
            if ((kDown & HidNpadButton_Down) && selected + 1 < n) selected++;
            if (kDown & HidNpadButton_A) {
                out = models[selected].path;
                return selected;
            }
        }
        return -1;
    }

    std::string generate(llama_context* ctx, const llama_vocab* vocab, llama_sampler* smpl,
                         const std::string& prompt, PadState* pad) {
        std::vector<llama_token> tokens = tokenize(vocab, prompt, true);
        for (size_t b = 0; b < tokens.size(); b += 256) {
            size_t chunk_len = std::min<size_t>(tokens.size() - b, 256);
            llama_batch batch = llama_batch_get_one(tokens.data() + b, (int32_t)chunk_len);
            if (llama_decode(ctx, batch) != 0) return "[decode error]";
        }

        std::string full_response;
        int n_cur = (int)tokens.size();
        for (int i = 0; i < N_PREDICT; ++i) {
            llama_token new_tok = llama_sampler_sample(smpl, ctx, -1);
            if (llama_vocab_is_eog(vocab, new_tok)) break;

            std::string piece = token_to_text(vocab, new_tok);
            full_response += piece;
            g_view.append(piece);
            g_view.render();

            llama_batch batch = llama_batch_get_one(&new_tok, 1);
            if (llama_decode(ctx, batch) != 0) break;
            n_cur++;

            padUpdate(pad);
            if (padGetButtonsDown(pad) & HidNpadButton_B) break;
        }
        return full_response;
    }

    bool run_chat(PadState* pad, const std::string& model_path) {
        consoleClear();
        printf("Initializing llama backend...\n");
        printf("Loading: %s\n", model_path.c_str());
        consoleUpdate(NULL);

        llama_backend_init();
        llama_log_set(log_cb, nullptr);

        llama_model_params mp = llama_model_default_params();
        mp.load_mode = LLAMA_LOAD_MODE_NONE; // standard buffered fread into heap

        llama_model* model = llama_model_load_from_file(model_path.c_str(), mp);
        if (!model) {
            printf("\n\x1b[1;31m[ERROR] Failed to load model from file.\x1b[0m\n");
            printf("Path: %s\n", model_path.c_str());
            printf("Make sure the model fits within Switch RAM (<2.8GB) and is not corrupt.\n");
            wait_for_exit(pad);
            return false;
        }

        llama_context_params cp = llama_context_default_params();
        cp.n_ctx = N_CTX;
        cp.n_batch = 256;
        cp.n_ubatch = 64;
        cp.n_threads = N_THREADS;
        cp.n_threads_batch = N_THREADS;

        llama_context* ctx = llama_init_from_model(model, cp);
        if (!ctx) {
            llama_model_free(model);
            printf("\n\x1b[1;31m[ERROR] Failed to create llama context\x1b[0m\n");
            wait_for_exit(pad);
            return false;
        }

        const llama_vocab* vocab = llama_model_get_vocab(model);
        llama_sampler* smpl = llama_sampler_chain_init(llama_sampler_chain_default_params());
        llama_sampler_chain_add(smpl, llama_sampler_init_temp(0.7f));
        llama_sampler_chain_add(smpl, llama_sampler_init_dist(42));

        const char* tmpl = llama_model_chat_template(model, nullptr);
        std::vector<llama_chat_message> messages;
        std::vector<char> formatted(4096);
        int prev_len = 0;

        g_view.clear();
        g_view.append("\x1b[1;36m=== LlamaNX Local Chat ===\x1b[0m\n");
        g_view.append("\x1b[2;37mPress A to type a prompt, B to stop generation, + to exit\x1b[0m\n");
        g_view.render();

        while (appletMainLoop()) {
            padUpdate(pad);
            u64 kDown = padGetButtonsDown(pad);
            if (kDown & HidNpadButton_Plus) break;

            if (kDown & HidNpadButton_A) {
                std::string input;
                if (read_line("Enter your prompt", input) && !input.empty()) {
                    g_view.append("\n\x1b[1;33mYou:\x1b[0m " + input + "\n\x1b[1;36mAI:\x1b[0m ");
                    g_view.render();

                    messages.push_back({ "user", strdup(input.c_str()) });

                    int new_len = llama_chat_apply_template(tmpl, messages.data(), messages.size(),
                                                            true, formatted.data(), formatted.size());
                    if (new_len > (int)formatted.size()) {
                        formatted.resize(new_len);
                        new_len = llama_chat_apply_template(tmpl, messages.data(), messages.size(),
                                                            true, formatted.data(), formatted.size());
                    }
                    if (new_len >= 0) {
                        std::string prompt(formatted.begin() + prev_len, formatted.begin() + new_len);
                        std::string reply = generate(ctx, vocab, smpl, prompt, pad);
                        messages.push_back({ "assistant", strdup(reply.c_str()) });
                        prev_len = llama_chat_apply_template(tmpl, messages.data(), messages.size(),
                                                             false, nullptr, 0);
                        if (prev_len < 0) prev_len = 0;
                    }
                    g_view.append("\n\x1b[2;37mPress A for next prompt, + to exit\x1b[0m\n");
                    g_view.render();
                }
            }
            consoleUpdate(NULL);
        }

        for (auto& m : messages) free((void*)m.content);
        llama_sampler_free(smpl);
        llama_free(ctx);
        llama_model_free(model);
        llama_backend_free();
        return true;
    }
}

int main(int argc, char* argv[]) {
    consoleInit(NULL);
    padConfigureInput(1, HidNpadStyleSet_NpadStandard);

    PadState pad;
    padInitializeDefault(&pad);

    if (!is_application_mode()) {
        printf("\x1b[1;31m[LlamaNX ERROR]\x1b[0m\n");
        printf("Applet mode detected (~450MB RAM limit).\n");
        printf("Please launch in Title Override mode (Hold R while launching a game).\n");
        wait_for_exit(&pad);
        consoleExit(NULL);
        return 0;
    }

    NetworkManager::init();

    while (appletMainLoop()) {
        consoleClear();
        u64 heap_mb = get_heap_total_bytes() / (1024 * 1024);
        std::string ip = NetworkManager::getLocalIP();

        printf("\x1b[1;35m====================================================\x1b[0m\n");
        printf("\x1b[1;37m                   LLAMANX v1.1.7                   \x1b[0m\n");
        printf("\x1b[2;37m      Forked from ChatNX by Souldbminer             \x1b[0m\n");
        printf("\x1b[1;35m====================================================\x1b[0m\n\n");
        printf("Available App Heap: \x1b[1;32m%llu MB\x1b[0m\n", (unsigned long long)heap_mb);
        printf("Network IP:         \x1b[1;33m%s\x1b[0m\n\n", ip.c_str());
        printf("Select Operation Mode:\n\n");
        printf("  \x1b[1;32m[A]\x1b[0m  Local Interactive Chat (On-Screen Keyboard)\n");
        printf("  \x1b[1;33m[Y]\x1b[0m  HTTP Server API (:8080 - Web UI & Remote)\n");
        printf("  \x1b[1;36m[X]\x1b[0m  Distributed RPC Node (:50052 - GGML PC Offload)\n\n");
        printf("  \x1b[1;31m[+]\x1b[0m  Exit to Homebrew Launcher\n\n");
        printf("----------------------------------------------------\n");
        consoleUpdate(NULL);

        padUpdate(&pad);
        u64 kDown = padGetButtonsDown(&pad);

        if (kDown & HidNpadButton_Plus) {
            break;
        }

        if (kDown & HidNpadButton_A) {
            std::vector<ModelEntry> models = scan_all_models();
            if (models.empty()) {
                consoleClear();
                printf("\x1b[1;31mNo GGUF models found!\x1b[0m\n\n");
                printf("Please copy .gguf files to:\n");
                printf("  sdmc:/switch/llamanx/\n");
                printf("  or sdmc:/switch/chatnx/\n");
                wait_for_exit(&pad);
                continue;
            }
            std::string model_path;
            if (select_model(&pad, models, model_path) >= 0) {
                run_chat(&pad, model_path);
            }
        }

        if (kDown & HidNpadButton_Y) {
            std::vector<ModelEntry> models = scan_all_models();
            std::string initial_model = models.empty() ? "" : models[0].path;
            run_http_server_mode(&pad, initial_model, 8080);
        }

        if (kDown & HidNpadButton_X) {
            run_rpc_server_mode(&pad, ip, 50052);
        }
    }

    NetworkManager::exit();
    consoleExit(NULL);
    return 0;
}
