param(
  [ValidateSet("local","multiarch")]
  [string]$Mode = "local",
  [string]$ImageName = "xavier/asr-agent",
  [string]$Tag = "dev",
  [switch]$WithNemo = $true
)

$ErrorActionPreference = "Stop"

Write-Host "Building image $($ImageName):$Tag (mode: $Mode)..." -ForegroundColor Cyan

# Ensure/reuse buildx builder
try {
  docker buildx inspect asr-builder *> $null
  docker buildx use asr-builder
} catch {
  docker buildx create --name asr-builder --use *> $null
}

# Build-args (as separate args)
$BuildArgs = @()
if ($WithNemo.IsPresent -and $WithNemo) {
  $BuildArgs += @("--build-arg", "WITH_NEMO=1")
  Write-Host "NeMo/TitaNet-S will be installed." -ForegroundColor Yellow
} else {
  Write-Host "NeMo/TitaNet-S disabled for this build." -ForegroundColor Yellow
}

$Dockerfile = "docker/Dockerfile"

if ($Mode -eq "local") {
  # IMPORTANT: on force TARGETARCH=amd64 pour le build local
  $BuildArgs += @("--build-arg", "TARGETARCH=amd64")

  docker buildx build `
    --builder asr-builder `
    --platform linux/amd64 `
    @BuildArgs `
    -t "$ImageName`:$Tag" `
    -f $Dockerfile `
    --load `
    .
  Write-Host "Local image loaded: $($ImageName):$Tag" -ForegroundColor Green

} elseif ($Mode -eq "multiarch") {
  # Laisser buildx injecter automatiquement TARGETARCH pour chaque plate-forme
  docker buildx build `
    --builder asr-builder `
    --platform linux/amd64,linux/arm64 `
    @BuildArgs `
    -t "$ImageName`:$Tag" `
    -f $Dockerfile `
    --push `
    .
  Write-Host "Multi-arch manifest pushed: $($ImageName):$Tag" -ForegroundColor Green

} else {
  throw "Unknown mode: $Mode (use 'local' or 'multiarch')"
}
