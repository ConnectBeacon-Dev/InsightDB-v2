# ==============================
# Setup Python Environment
# ==============================

# Base folder = current working directory
$baseFolder = Get-Location
Write-Host "Base folder set to:" $baseFolder -ForegroundColor Yellow

# Create venv if not exists
if (-Not (Test-Path "$baseFolder\venv")) {
    Write-Host "Creating virtual environment..." -ForegroundColor Green
    python -m venv "$baseFolder\venv"
} else {
    Write-Host "Virtual environment already exists." -ForegroundColor Cyan
}

# Activate venv
$venvActivate = "$baseFolder\venv\Scripts\Activate.ps1"
if (Test-Path $venvActivate) {
    Write-Host "Activating virtual environment..." -ForegroundColor Green
    & $venvActivate
} else {
    Write-Host "Activation script not found!" -ForegroundColor Red
    exit 1
}

# Install requirements - check both root and flat-file-approach directories
$requirementsFound = $false

if (Test-Path "$baseFolder\requirements.txt") {
    Write-Host "Installing from root requirements.txt..." -ForegroundColor Green
    pip install -r "$baseFolder\requirements.txt"
    $requirementsFound = $true
} elseif (Test-Path "$baseFolder\flat-file-approach\requirements.txt") {
    Write-Host "Installing from flat-file-approach\requirements.txt..." -ForegroundColor Green
    pip install -r "$baseFolder\flat-file-approach\requirements.txt"
    $requirementsFound = $true
} else {
    Write-Host "No requirements.txt found in root or flat-file-approach directory." -ForegroundColor Yellow
}

# Install essential packages if requirements.txt wasn't found or to ensure they're available
if (-Not $requirementsFound) {
    Write-Host "Installing essential packages manually..." -ForegroundColor Green
    pip install pandas numpy scikit-learn flask sentence-transformers huggingface_hub torch transformers pyarrow spacy
} else {
    # Ensure critical packages are installed even if requirements.txt exists
    Write-Host "Ensuring critical packages are installed..." -ForegroundColor Green
    pip install --upgrade pandas huggingface_hub pyarrow
}

# Install spaCy English model
Write-Host "Installing spaCy English model..." -ForegroundColor Green
python -m spacy download en_core_web_sm

Write-Host "`n✅ Setup complete! Use '.\venv\Scripts\Activate.ps1' to activate later." -ForegroundColor Cyan
