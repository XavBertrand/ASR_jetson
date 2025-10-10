# TensorRT-LLM Deployment on Jetson Orin Nano

Complete guide to deploy **Qwen 2.5 1.5B** with TensorRT-LLM on your Jetson.

## 📋 Prerequisites

- Jetson Orin Nano with JetPack 6.0+ (r36.2.0)  
- At least 8 GB of available RAM  
- At least 20 GB of free disk space  
- Docker and nvidia-container-runtime installed  

---

## 🚀 Quick Deployment (recommended)

### Option A: Using a pre-built engine

If you already have a TensorRT-LLM engine:

```bash
# 1. Clone the repository
cd ~/ASR_Agent

# 2. Create directories for engines
mkdir -p volumes/trtllm-engines volumes/trtllm-checkpoints

# 3. Copy your pre-built engine (if available)
# cp -r /path/to/qwen2.5-1.5b-engine volumes/trtllm-engines/qwen2.5-1.5b

# 4. Launch the services
docker-compose -f docker-compose.jetson.yml up -d

# 5. Check status
docker-compose -f docker-compose.jetson.yml ps
docker-compose -f docker-compose.jetson.yml logs -f tensorrt-llm
```

---

### Option B: Build the engine on the fly

If you don’t have a pre-built engine yet:

```bash
# 1. Start only the TensorRT-LLM service
docker-compose -f docker-compose.jetson.yml up -d tensorrt-llm

# 2. Enter the container
docker exec -it trtllm-qwen bash

# 3. Build the engine (takes 15–30 minutes)
/app/build_engine.sh

# 4. Verify that the engine was created
ls -lh /workspace/trt_engines/qwen2.5-1.5b/

# 5. Restart the service
exit
docker-compose -f docker-compose.jetson.yml restart tensorrt-llm

# 6. Launch the ASR service
docker-compose -f docker-compose.jetson.yml up -d asr-pipeline
```

---

## 🧪 Test the Installation

### Test TensorRT-LLM server

```bash
# Health check
curl http://localhost:8001/health

# Generation test
curl -X POST http://localhost:8001/v1/chat/completions   -H "Content-Type: application/json"   -d '{
    "model": "qwen2.5-1.5b-instruct",
    "messages": [
      {
        "role": "system",
        "content": "You are a text-correction assistant."
      },
      {
        "role": "user",
        "content": "Fix this text: bonjour  ,je  mappelle  xavier"
      }
    ],
    "temperature": 0.1,
    "max_tokens": 100
  }'
```

### Test your ASR pipeline

```bash
# Your ASR pipeline should now automatically use TensorRT-LLM
# Check logs
docker-compose -f docker-compose.jetson.yml logs -f asr-pipeline
```

---

## 📊 Expected Performance

On Jetson Orin Nano (8 GB):

| Metric | Value |
|--------|--------|
| **Latency (100-token prompt)** | ~200–400 ms |
| **Throughput** | ~30–50 tokens/s |
| **RAM usage** | ~2–3 GB |
| **VRAM usage** | ~1.5–2 GB |
| **Engine build time** | 15–30 min |

---

## 🔧 Advanced Configuration

### Optimize for your use case

Edit `build_qwen_trt_engine.sh`:

```bash
# For short texts (short transcriptions)
MAX_INPUT_LEN=512
MAX_OUTPUT_LEN=256

# For longer texts (long transcriptions)
MAX_INPUT_LEN=4096
MAX_OUTPUT_LEN=1024

# Batch size (if processing multiple files)
MAX_BATCH_SIZE=8
```

### Environment variables

In `docker-compose.jetson.yml`, you can adjust:

```yaml
environment:
  - LLM_ENDPOINT=http://tensorrt-llm:8000
  - LLM_MODEL=qwen2.5-1.5b-instruct
  - LLM_API_KEY=  # Optional if you add authentication
```

---

## 🐛 Troubleshooting

### Engine fails to build

```bash
# Check logs
docker-compose -f docker-compose.jetson.yml logs tensorrt-llm

# Check disk space
df -h

# Check available RAM
free -h
```

### “CUDA out of memory” error

```bash
# Reduce batch size
# In build_qwen_trt_engine.sh:
MAX_BATCH_SIZE=1
```

### Server won’t start

```bash
# Check if the engine exists
docker exec -it trtllm-qwen ls -lh /workspace/trt_engines/qwen2.5-1.5b/

# Check permissions
docker exec -it trtllm-qwen chmod -R 755 /workspace/trt_engines/
```

---

## 📦 File Structure

```
~/ASR_Agent/
├── docker/
│   ├── Dockerfile.jetson            # Your ASR image
│   └── Dockerfile.tensorrt-llm      # TensorRT-LLM image
├── scripts/
│   ├── build_qwen_trt_engine.sh     # Build script
│   └── trtllm_server.py             # API server
├── docker-compose.jetson.yml        # Orchestration
└── volumes/                         # Persistent data
    ├── trtllm-engines/              # TRT engines (no rebuild needed)
    └── trtllm-checkpoints/          # Intermediate checkpoints
```

---

## 🔄 Updating

To upgrade to a newer version of Qwen:

```bash
# 1. Stop services
docker-compose -f docker-compose.jetson.yml down

# 2. Remove the old engine
rm -rf volumes/trtllm-engines/qwen2.5-1.5b/*

# 3. Update MODEL_NAME in build_qwen_trt_engine.sh
# MODEL_NAME="Qwen/Qwen2.5-3B-Instruct"  # Example for 3B

# 4. Rebuild and restart
docker-compose -f docker-compose.jetson.yml up -d --build
```

---

## 💾 Engine Backup

Building the TensorRT engine is time-consuming (15–30 min). Back it up:

```bash
# Create an archive of the engine
tar -czf qwen2.5-1.5b-trt-engine.tar.gz volumes/trtllm-engines/qwen2.5-1.5b/

# Restore on another machine
tar -xzf qwen2.5-1.5b-trt-engine.tar.gz -C volumes/trtllm-engines/
```

---

## 📚 Resources

- [TensorRT-LLM Documentation](https://github.com/NVIDIA/TensorRT-LLM)  
- [Qwen 2.5 Model Card](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct)  
- [Jetson AI Lab](https://www.jetson-ai-lab.com/)