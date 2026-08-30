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

$env:UV_CACHE_DIR = Join-Path $LabRoot ".cache\uv"
$env:UV_PYTHON_INSTALL_DIR = Join-Path $LabRoot ".cache\uv\python"
$env:HF_HOME = Join-Path $LabRoot ".cache\huggingface"
$env:TORCH_HOME = Join-Path $LabRoot ".cache\torch"
$env:TOKENIZERS_PARALLELISM = "false"
$env:OMP_NUM_THREADS = "1"
$env:MKL_NUM_THREADS = "1"
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

if (-not (Test-Path -LiteralPath $Uv)) {
    throw "Repository environment is missing. Run: .\lab.ps1 bootstrap"
}

$CliArgs = @($Command)
if ($Profile) {
    $CliArgs += "--profile"
    $CliArgs += $Profile
}
$CliArgs += $RemainingArgs
& $Uv run --frozen python -m osrlab @CliArgs
exit $LASTEXITCODE
