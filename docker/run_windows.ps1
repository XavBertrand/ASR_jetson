# Run the ASR Agent container with GPU by default on Windows (Docker Desktop + WSL2)
param(
  [string]$ImageName = "xavier/asr-agent",
  [string]$Tag = "latest",
  [string]$Name = "asr-agent"
)

$PwdPath = (Get-Location).Path
$HfCache = "$HOME\.cache\huggingface"

Write-Host "Running $ImageName:$Tag"
Write-Host "Using GPU flag: --gpus all"
Write-Host "Mount: $PwdPath -> /app"

docker run --rm -it `
  --gpus all `
  --name $Name `
  -v "$PwdPath:/app" `
  -v "$HfCache:/root/.cache/huggingface" `
  -w /app `
  "$ImageName`:$Tag" `
  bash
