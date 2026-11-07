# LlamaNX

**LlamaNX** is a native Nintendo Switch (Horizon OS / `libnx`) port of [`llama.cpp`](https://github.com/ggml-org/llama.cpp) supporting on-device interactive chat, local network OpenAI-compatible HTTP serving, and distributed GGML RPC cluster offloading.

---

### Notice & Attribution
> **Credits:** LlamaNX is an extended fork of [ChatNX](https://github.com/souldbminerr/ChatNX) by **Souldbminer**.
> Original ChatNX provided the pioneering foundation for compiling and running `llama.cpp` natively under Horizon OS with `devkitA64` and `libnx`.
> LlamaNX extends ChatNX by adding:
> - **Distributed RPC Node Mode (`ggml-rpc-server`)**: Use your Switch as a network-attached acceleration/RAM offload node for a host PC running `llama.cpp`.
> - **HTTP Server API Mode (`llama-server`)**: Query models running on your Switch over LAN via standard OpenAI-compatible REST endpoints (`/v1/chat/completions`).
> - **Multi-mode selector launcher**: Switch seamlessly between Local Chat, HTTP Server, and RPC Node directly from the Switch UI.

---

## Features
- **Local Chat (A)**: Interactive on-device chat with the Switch software keyboard (`swkbd`) and on-screen text rendering.
- **HTTP Server (Y)**: Hosts an HTTP API server on port `8080`. Send prompts from Open-WebUI, curl, or mobile devices to your Switch.
- **Distributed RPC Node (X)**: Hosts GGML RPC on port `50052`. Offload layers from your PC to the Switch's 3.2 GB RAM and Tegra X1 CPU.

## Requirements
- Nintendo Switch running Custom Firmware (**Atmosphere**).
- **Title Override mode is MANDATORY:** Hold **`R`** while launching any installed game title to grant homebrew full access to ~3.2 GB of RAM. (Launching via the Album applet will crash due to the ~450 MB memory limit).
- GGUF models placed in `sdmc:/switch/llamanx/` or `sdmc:/switch/chatnx/` (Q4_K_M or smaller recommended, 0.5B to 1.5B parameters such as Qwen2.5-0.5B, SmolLM-1.7B, or Llama-3.2-1B).

## Usage
### 1. Local Chat
Place `.gguf` files into `/switch/llamanx/` on your SD card. Launch LlamaNX, select your model, and chat directly using the touchscreen/Joy-Con keyboard.

### 2. Distributed RPC Node
1. Launch LlamaNX and press **(X)** to start the RPC Server.
2. Note the IP address displayed on your Switch screen (e.g. `192.168.1.150`).
3. On your PC, run `llama.cpp` with the `--rpc` flag to offload model layers to the Switch:
   ```bash
   llama-cli -m model.gguf -p "Hello!" --rpc 192.168.1.150:50052 -ngl 4
   ```

### 3. HTTP Server Mode
1. Launch LlamaNX and press **(Y)** to start the HTTP Server with your chosen model.
2. Send completions from any LAN device:
   ```bash
   curl http://192.168.1.150:8080/v1/chat/completions \
     -H "Content-Type: application/json" \
     -d '{"messages": [{"role": "user", "content": "Hello!"}]}'
   ```

## Building
To build `LlamaNX.nro`, use Docker with `devkitpro/devkita64`:
```powershell
./build_docker.ps1
```
Or manually with `devkitPro`:
```bash
./scripts/build-llama.sh
make -j$(nproc)
```

## License
MIT License (see `LICENSE`). Based on ChatNX by Souldbminer and `llama.cpp` by Georgi Gerganov & contributors.
