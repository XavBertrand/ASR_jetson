# Déploiement TensorRT-LLM sur Jetson Orin Nano

Guide complet pour déployer Qwen 2.5 1.5B avec TensorRT-LLM sur votre Jetson.

## 📋 Prérequis

- Jetson Orin Nano avec JetPack 6.0+ (r36.2.0)
- Au moins 8 GB RAM disponible
- Au moins 20 GB d'espace disque libre
- Docker et nvidia-container-runtime installés

## 🚀 Déploiement rapide (recommandé)

### Option A : Avec moteur pré-buildé

Si vous avez déjà un moteur TensorRT-LLM :

```bash
# 1. Cloner le repo
cd ~/ASR_Agent

# 2. Créer les répertoires pour les moteurs
mkdir -p volumes/trtllm-engines volumes/trtllm-checkpoints

# 3. Copier votre moteur pré-buildé (si disponible)
# cp -r /path/to/qwen2.5-1.5b-engine volumes/trtllm-engines/qwen2.5-1.5b

# 4. Lancer les services
docker-compose -f docker-compose.jetson.yml up -d

# 5. Vérifier le status
docker-compose -f docker-compose.jetson.yml ps
docker-compose -f docker-compose.jetson.yml logs -f tensorrt-llm
```

### Option B : Build du moteur à la volée

Si vous n'avez pas encore de moteur :

```bash
# 1. Lancer uniquement le service TensorRT-LLM
docker-compose -f docker-compose.jetson.yml up -d tensorrt-llm

# 2. Entrer dans le container
docker exec -it trtllm-qwen bash

# 3. Builder le moteur (15-30 minutes)
/app/build_engine.sh

# 4. Vérifier que le moteur est créé
ls -lh /workspace/trt_engines/qwen2.5-1.5b/

# 5. Redémarrer le service
exit
docker-compose -f docker-compose.jetson.yml restart tensorrt-llm

# 6. Lancer le service ASR
docker-compose -f docker-compose.jetson.yml up -d asr-pipeline
```

## 🧪 Tester l'installation

### Test du serveur TensorRT-LLM

```bash
# Health check
curl http://localhost:8001/health

# Test de génération
curl -X POST http://localhost:8001/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen2.5-1.5b-instruct",
    "messages": [
      {
        "role": "system",
        "content": "Tu es un assistant de correction de texte."
      },
      {
        "role": "user",
        "content": "Corrige ce texte: bonjour  ,je  mappelle  xavier"
      }
    ],
    "temperature": 0.1,
    "max_tokens": 100
  }'
```

### Test de votre pipeline ASR

```bash
# Votre pipeline ASR devrait maintenant utiliser automatiquement TensorRT-LLM
# Vérifier les logs
docker-compose -f docker-compose.jetson.yml logs -f asr-pipeline
```

## 📊 Performances attendues

Sur Jetson Orin Nano (8 GB) :

| Métrique | Valeur |
|----------|--------|
| **Latence (prompt 100 tokens)** | ~200-400ms |
| **Throughput** | ~30-50 tokens/s |
| **RAM utilisée** | ~2-3 GB |
| **VRAM utilisée** | ~1.5-2 GB |
| **Temps de build moteur** | 15-30 min |

## 🔧 Configuration avancée

### Optimiser pour votre cas d'usage

Modifiez `build_qwen_trt_engine.sh` :

```bash
# Pour des textes plus courts (transcriptions courtes)
MAX_INPUT_LEN=512
MAX_OUTPUT_LEN=256

# Pour des textes plus longs (transcriptions longues)
MAX_INPUT_LEN=4096
MAX_OUTPUT_LEN=1024

# Batch size (si vous traitez plusieurs fichiers)
MAX_BATCH_SIZE=8
```

### Variables d'environnement

Dans `docker-compose.jetson.yml`, vous pouvez ajuster :

```yaml
environment:
  - LLM_ENDPOINT=http://tensorrt-llm:8000
  - LLM_MODEL=qwen2.5-1.5b-instruct
  - LLM_API_KEY=  # Optionnel si vous ajoutez de l'auth
```

## 🐛 Dépannage

### Le moteur ne se build pas

```bash
# Vérifier les logs
docker-compose -f docker-compose.jetson.yml logs tensorrt-llm

# Vérifier l'espace disque
df -h

# Vérifier la RAM disponible
free -h
```

### Erreur "CUDA out of memory"

```bash
# Réduire le batch size
# Dans build_qwen_trt_engine.sh :
MAX_BATCH_SIZE=1
```

### Le serveur ne démarre pas

```bash
# Vérifier que le moteur existe
docker exec -it trtllm-qwen ls -lh /workspace/trt_engines/qwen2.5-1.5b/

# Vérifier les permissions
docker exec -it trtllm-qwen chmod -R 755 /workspace/trt_engines/
```

## 📦 Structure des fichiers

```
~/ASR_Agent/
├── docker/
│   ├── Dockerfile.jetson            # Votre image ASR
│   └── Dockerfile.tensorrt-llm      # Image TensorRT-LLM
├── scripts/
│   ├── build_qwen_trt_engine.sh     # Script de build
│   └── trtllm_server.py             # Serveur API
├── docker-compose.jetson.yml        # Orchestration
└── volumes/                         # Persistent data
    ├── trtllm-engines/              # Moteurs TRT (rebuild pas nécessaire)
    └── trtllm-checkpoints/          # Checkpoints intermédiaires
```

## 🔄 Mise à jour

Pour mettre à jour vers une nouvelle version de Qwen :

```bash
# 1. Arrêter les services
docker-compose -f docker-compose.jetson.yml down

# 2. Supprimer l'ancien moteur
rm -rf volumes/trtllm-engines/qwen2.5-1.5b/*

# 3. Modifier MODEL_NAME dans build_qwen_trt_engine.sh
# MODEL_NAME="Qwen/Qwen2.5-3B-Instruct"  # Exemple pour 3B

# 4. Rebuild et redémarrer
docker-compose -f docker-compose.jetson.yml up -d --build
```

## 💾 Sauvegarde du moteur

Le moteur TensorRT est lourd à rebuilder (15-30 min). Sauvegardez-le :

```bash
# Créer une archive du moteur
tar -czf qwen2.5-1.5b-trt-engine.tar.gz volumes/trtllm-engines/qwen2.5-1.5b/

# Restaurer sur une autre machine
tar -xzf qwen2.5-1.5b-trt-engine.tar.gz -C volumes/trtllm-engines/
```

## 📚 Ressources

- [TensorRT-LLM Documentation](https://github.com/NVIDIA/TensorRT-LLM)
- [Qwen 2.5 Model Card](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct)
- [Jetson AI Lab](https://www.jetson-ai-lab.com/)