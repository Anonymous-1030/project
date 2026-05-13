# PowerShell script to run vLLM in WSL with ProSE-X 2.0 metrics
#
# Usage:
#   .\run_vllm_wsl.ps1 -Model "microsoft/Phi-3-mini-4k-instruct" -ContextLength 4096
#

param(
    [string]$Model = "microsoft/Phi-3-mini-4k-instruct",
    [int]$ContextLength = 4096,
    [int]$BudgetRatioPercent = 10,
    [string]$WorkloadType = "passkey",
    [string]$OutputDir = "outputs/vllm_wsl",
    [switch]$UseProSEv2 = $true,
    [switch]$DetailedMetrics = $true
)

$ErrorActionPreference = "Stop"

Write-Host "==========================================" -ForegroundColor Cyan
Write-Host "ProSE-X 2.0 vLLM Runner for WSL" -ForegroundColor Cyan
Write-Host "==========================================" -ForegroundColor Cyan
Write-Host ""

# Check if WSL is available
try {
    $wslCheck = wsl --version 2>&1
    Write-Host "✓ WSL is available" -ForegroundColor Green
} catch {
    Write-Host "✗ WSL not found. Please install WSL first." -ForegroundColor Red
    exit 1
}

# Convert Windows path to WSL path
$windowsPath = (Get-Location).Path
$wslPath = wsl wslpath -a "$windowsPath"
Write-Host "Working directory: $windowsPath -> $wslPath (WSL)" -ForegroundColor Gray

# Create output directory
New-Item -ItemType Directory -Force -Path $OutputDir | Out-Null
$wslOutputDir = "$wslPath/$OutputDir"

# Build the command
$pythonScript = "prose_v2/src/runners/run_vllm_metrics.py"
$wslScriptPath = "$wslPath/$pythonScript"

$args = @(
    "--model", $Model
    "--context-length", $ContextLength
    "--budget-ratio", ($BudgetRatioPercent / 100)
    "--workload-type", $WorkloadType
    "--output-dir", $wslOutputDir
)

if ($UseProSEv2) {
    $args += "--use-prose-v2"
}

if ($DetailedMetrics) {
    $args += "--detailed-metrics"
}

Write-Host ""
Write-Host "Running Configuration:" -ForegroundColor Yellow
Write-Host "  Model: $Model" -ForegroundColor White
Write-Host "  Context Length: $ContextLength" -ForegroundColor White
Write-Host "  Budget Ratio: $BudgetRatioPercent%" -ForegroundColor White
Write-Host "  Workload Type: $WorkloadType" -ForegroundColor White
Write-Host "  Use ProSE-X v2: $UseProSEv2" -ForegroundColor White
Write-Host "  Detailed Metrics: $DetailedMetrics" -ForegroundColor White
Write-Host ""

# Activate conda and run
$command = @"
cd $wslPath && 
source ~/miniconda3/etc/profile.d/conda.sh && 
conda activate vllm && 
python $wslScriptPath $($args -join ' ')
"@

Write-Host "Executing in WSL..." -ForegroundColor Yellow
Write-Host "Command: $command" -ForegroundColor DarkGray
Write-Host ""

try {
    wsl bash -c "$command"
    
    Write-Host ""
    Write-Host "==========================================" -ForegroundColor Green
    Write-Host "Run completed successfully!" -ForegroundColor Green
    Write-Host "Output saved to: $OutputDir" -ForegroundColor Green
    Write-Host "==========================================" -ForegroundColor Green
} catch {
    Write-Host ""
    Write-Host "==========================================" -ForegroundColor Red
    Write-Host "Run failed with error:" -ForegroundColor Red
    Write-Host $_ -ForegroundColor Red
    Write-Host "==========================================" -ForegroundColor Red
    exit 1
}
