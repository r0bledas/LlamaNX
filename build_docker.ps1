# Quick Docker Build Script for LlamaNX
$ErrorActionPreference = "Stop"

Write-Host "Checking Docker..." -ForegroundColor Cyan
if (-not (Get-Command docker -ErrorAction SilentlyContinue)) {
    Write-Error "Docker is required to cross-compile for Nintendo Switch (devkitA64)."
    exit 1
}

$WORKSPACE = (Get-Location).Path -replace '\\', '/'
Write-Host "Building LlamaNX inside devkitpro/devkita64 container..." -ForegroundColor Green

docker run --rm -v "${WORKSPACE}:/workspace" -w /workspace devkitpro/devkita64 /bin/bash -c @"
set -e
git config --global --add safe.directory /workspace
git config --global --add safe.directory /workspace/vendor/llama.cpp

echo '==> 1. Building llama.cpp static libraries for Switch...'
chmod +x scripts/build-llama.sh
./scripts/build-llama.sh

echo '==> 2. Compiling LlamaNX homebrew (.nro)...'
make clean
make -j\$(nproc)

echo '==> Build Complete!'
ls -lh LlamaNX.nro
"@

if ($LASTEXITCODE -eq 0) {
    Write-Host "`nSuccessfully compiled LlamaNX.nro!" -ForegroundColor Green
    Write-Host "Copy LlamaNX.nro to your Switch SD card under /switch/LlamaNX/LlamaNX.nro" -ForegroundColor Yellow
} else {
    Write-Error "Build failed. Check the container output above."
}
