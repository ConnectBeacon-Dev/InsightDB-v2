$workFolder = (Get-Location).Path
$tools  = Join-Path $workFolder "tools"

# --- Install llama-cpp-python (MinGW via w64devkit) ---
Write-Host "Installing llama-cpp-python prerequisites..." -ForegroundColor Green
$ErrorActionPreference = 'Stop'

# tools directory lives inside the extracted repo
New-Item -ItemType Directory -Force -Path $tools | Out-Null

# Activate venv if not already active (venv was created under $workFolder)
$activate = Join-Path $workFolder "venv\Scripts\Activate.ps1"
if (Test-Path $activate) {
  Write-Host "Activating virtual environment..." -ForegroundColor Green
  & $activate
} else {
  Write-Host "Virtual env not found at: $activate" -ForegroundColor Red
  exit 1
}

# --- Download and extract w64devkit ---
Set-Location $tools
$w64Exe = Join-Path $tools "w64devkit-x64-2.3.0.7z.exe"
if (-not (Test-Path $w64Exe)) {
  Write-Host "Downloading w64devkit..." -ForegroundColor Green
  Invoke-WebRequest `
    -Uri "https://github.com/skeeto/w64devkit/releases/download/v2.3.0/w64devkit-x64-2.3.0.7z.exe" `
    -OutFile $w64Exe
} else {
  Write-Host "w64devkit archive already present, skipping download." -ForegroundColor Yellow
}

# Self-extractor supports -y and -o<dir>
Write-Host "Extracting w64devkit..." -ForegroundColor Green
Start-Process -FilePath $w64Exe -ArgumentList "-y -o$tools\" -Wait -NoNewWindow

$w64Bin = Join-Path $tools "w64devkit\bin"
if (-not (Test-Path $w64Bin)) {
  Write-Host "w64devkit extraction failed. Expected: $w64Bin" -ForegroundColor Red
  exit 1
}

Write-Host "Toolchain files:" -ForegroundColor Green
Get-ChildItem $w64Bin

# --- Toolchain & CMake environment for pip build ---
$env:PATH = "$w64Bin;$env:PATH"
$gcc   = Join-Path $w64Bin "gcc.exe"
$gxx   = Join-Path $w64Bin "g++.exe"
$make  = Join-Path $w64Bin "make.exe"

if (-not (Test-Path $gcc)) { Write-Host "gcc not found at $gcc" -ForegroundColor Red; exit 1 }
if (-not (Test-Path $gxx)) { Write-Host "g++ not found at $gxx" -ForegroundColor Red; exit 1 }
if (-not (Test-Path $make)) { Write-Host "make not found at $make" -ForegroundColor Red; exit 1 }

# Let CMake/pip know which compilers/generator to use
$env:CC = $gcc
$env:CXX = $gxx
$env:CMAKE_GENERATOR = "MinGW Makefiles"
$env:CMAKE_MAKE_PROGRAM = $make

# Helpful CMake args; OPENBLAS speeds up; adjust if you don’t want it
$env:CMAKE_ARGS = @(
  "-DGGML_OPENBLAS=on"
  "-DCMAKE_C_COMPILER=`"$gcc`""
  "-DCMAKE_CXX_COMPILER=`"$gxx`""
  "-DCMAKE_MAKE_PROGRAM=`"$make`""
) -join " "

# Force source build (avoid prebuilt wheels that might target MSVC)
$env:FORCE_CMAKE = "1"
# If you are CPU-only, keep CUBLAS off (default is off on Windows)
$env:LLAMA_CUBLAS = "0"

Write-Host "Assembler version:" -ForegroundColor Green
& (Join-Path $w64Bin "as.exe") --version

# --- Install llama-cpp-python ---
Write-Host "Installing llama-cpp-python via pip (this may take a while)..." -ForegroundColor Green
pip install --force-reinstall llama-cpp-python

Write-Host "Verifying installation..." -ForegroundColor Green
python -c "import llama_cpp; print('llama-cpp-python installed OK')"
