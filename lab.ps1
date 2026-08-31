[CmdletBinding()]
param(
    [Parameter(Position = 0)]
    [string]$Command = "verify",
    [string]$Profile,
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$RemainingArgs
)

$ErrorActionPreference = "Stop"
$LabRoot = $PSScriptRoot
$BootstrapPython = Join-Path $LabRoot ".tools\bootstrap\Scripts\python.exe"
$Uv = Join-Path $LabRoot ".tools\bootstrap\Scripts\uv.exe"
$GpuVenv = Join-Path $LabRoot ".venv-gpu"
$GpuPython = Join-Path $GpuVenv "Scripts\python.exe"
$GpuLock = Join-Path $LabRoot "configs\gpu-requirements.lock"

$env:UV_CACHE_DIR = Join-Path $LabRoot ".cache\uv"
$env:UV_PYTHON_INSTALL_DIR = Join-Path $LabRoot ".cache\uv\python"
$env:HF_HOME = Join-Path $LabRoot ".cache\huggingface"
$env:TORCH_HOME = Join-Path $LabRoot ".cache\torch"
$env:TOKENIZERS_PARALLELISM = "false"
$env:OMP_NUM_THREADS = "1"
$env:MKL_NUM_THREADS = "1"
$env:CUBLAS_WORKSPACE_CONFIG = ":4096:8"
$env:TEMP = Join-Path $LabRoot ".cache\tmp"
$env:TMP = $env:TEMP
New-Item -ItemType Directory -Force -Path $env:TEMP | Out-Null

if ($Command -eq "bootstrap") {
    if (-not (Test-Path -LiteralPath $BootstrapPython)) {
        python -m venv (Join-Path $LabRoot ".tools\bootstrap")
    }
    & $BootstrapPython -m pip install --disable-pip-version-check "uv==0.8.14"
    & $Uv python install 3.10 --no-bin --no-registry
    & $Uv sync --extra dev --frozen
    exit $LASTEXITCODE
}

if ($Command -eq "gpu-bootstrap") {
    if (-not (Test-Path -LiteralPath $Uv)) {
        throw "Repository environment is missing. Run: .\lab.ps1 bootstrap"
    }
    & $Uv venv $GpuVenv --python 3.10 --clear
    if ($LASTEXITCODE) { exit $LASTEXITCODE }
    & $Uv pip sync --python $GpuPython --index "https://download.pytorch.org/whl/cu126" --index-strategy unsafe-best-match $GpuLock
    if ($LASTEXITCODE) { exit $LASTEXITCODE }
    & $Uv pip install --python $GpuPython --no-deps -e $LabRoot
    if ($LASTEXITCODE) { exit $LASTEXITCODE }
    & $GpuPython -c "import torch; assert torch.cuda.is_available(); print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0))"
    exit $LASTEXITCODE
}

if (-not (Test-Path -LiteralPath $Uv)) {
    throw "Repository environment is missing. Run: .\lab.ps1 bootstrap"
}

$CliArgs = @($Command)
if ($Profile) {
    $CliArgs += "--profile"
    $CliArgs += $Profile
}
$CliArgs += $RemainingArgs
if ($Command -in @("p5", "tune-e3", "tune-recall")) {
    if (-not (Test-Path -LiteralPath $GpuPython)) {
        throw "GPU environment is missing. Run: .\lab.ps1 gpu-bootstrap"
    }
    & $GpuPython -m osrlab @CliArgs
} else {
    & $Uv run --frozen python -m osrlab @CliArgs
}
exit $LASTEXITCODE
