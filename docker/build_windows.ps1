# Build multi-arch docker image for ASR Agent with NeMo/TitaNet-S

$ImageName = "xavier/asr-agent"
$Tag = "latest"

Write-Host "Building multi-arch image ($ImageName:$Tag) with NeMo (TitaNet-S)..."

# Ensure buildx builder exists
docker buildx create --use --name asr-builder 2>$null | Out-Null

docker buildx build `
  --platform linux/amd64,linux/arm64 `
  --build-arg WITH_NEMO=1 `
  -t "$ImageName`:$Tag" `
  -f docker/Dockerfile `
  .
