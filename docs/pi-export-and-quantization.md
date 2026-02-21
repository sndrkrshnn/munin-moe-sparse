# Pi Export + Quantization (GGUF)

## 1) Export checkpoint to GGUF

```bash
python scripts/export_to_gguf.py \
  --checkpoint artifacts/latest \
  --llama-cpp-dir ~/llama.cpp \
  --out artifacts/gguf/munin-moe-f16.gguf \
  --dtype f16
```

## 2) Quantize for Raspberry Pi 5 CPU

```bash
cd ~/llama.cpp
./build/bin/llama-quantize \
  /path/to/munin-moe-f16.gguf \
  /path/to/munin-moe-q4_k_m.gguf \
  Q4_K_M
```

Optional quality profile:
```bash
./build/bin/llama-quantize in.gguf out.gguf Q5_K_M
```

## 3) Benchmark on Pi

```bash
./build/bin/llama-bench -m /path/to/munin-moe-q4_k_m.gguf -p "hello" -n 128
```

Track:
- first-token latency
- tokens/sec
- peak RAM
- thermal throttling after sustained decode
