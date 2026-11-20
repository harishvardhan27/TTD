# run_queue.ps1 - the remaining compute queue, in dependency order.
#
#   powershell -ExecutionPolicy Bypass -File scripts\run_queue.ps1            # all stages
#   powershell -ExecutionPolicy Bypass -File scripts\run_queue.ps1 -From 3    # start at stage 3
#
# Every stage is resumable (finished variants/runs are skipped), so re-running
# after an interruption is always safe. The queue stops at the first failing
# stage so later stages never build on a broken result. Before each stage it
# waits until at least $MinFreeGB of RAM is available.
#
# Stages
#   1  DeiT-S test grid (configs/deit_small.yaml, suite core)          ~2.5 h left
#   2  Distillation selection + final runs (protocol section 7)         ~6-7 h
#   3  ViT-B confirmatory test grid (configs/vit_base.yaml, core)       ~6 h
#   4  CPU benchmarks: torch, ORT, RSS, layer microbenchmarks           ~1.5 h  (AC power required)
#   5  Analysis: tables and figures for every finished suite

param([int]$From = 1, [double]$MinFreeGB = 3.0)

$ErrorActionPreference = "Stop"
Set-Location (Split-Path $PSScriptRoot -Parent)
$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONWARNINGS = "ignore"
$log = "results\raw\queue.log"

function Log($msg) { $line = "[{0}] {1}" -f (Get-Date -Format "yyyy-MM-dd HH:mm:ss"), $msg; Write-Host $line; Add-Content $log $line }

function Wait-Memory {
    while ($true) {
        $free = (Get-CimInstance Win32_OperatingSystem).FreePhysicalMemory / 1MB
        if ($free -ge $MinFreeGB) { return }
        Log ("only {0:N1} GB free; waiting for >= {1} GB (close other apps)" -f $free, $MinFreeGB)
        Start-Sleep -Seconds 60
    }
}

function Stage($n, $name, [scriptblock]$body) {
    if ($n -lt $From) { return }
    Wait-Memory
    Log "=== stage ${n}: $name"
    & $body
    if ($LASTEXITCODE -ne 0) { Log "stage $n FAILED (exit $LASTEXITCODE); stopping queue"; exit $LASTEXITCODE }
    Log "=== stage $n done"
}

Stage 1 "DeiT-S core grid on v2_test" { python experiments/run.py --config configs/deit_small.yaml --suite core --split v2_test }
Stage 2 "distillation (selection + final)" { python experiments/distill_select.py --config configs/deit_small.yaml }
Stage 3 "ViT-B core grid on v2_test" { python experiments/run.py --config configs/vit_base.yaml --suite core --split v2_test }
Stage 4 "CPU benchmarks" {
    python experiments/benchmark.py --config configs/deit_small.yaml --what model --backend torch
    if ($LASTEXITCODE -eq 0) { python experiments/benchmark.py --config configs/deit_small.yaml --what model --backend ort }
    if ($LASTEXITCODE -eq 0) { python experiments/benchmark.py --config configs/deit_small.yaml --what rss }
    if ($LASTEXITCODE -eq 0) { python experiments/benchmark.py --what layers }
}
Stage 5 "analysis" {
    python experiments/analyze.py --model deit_small --split v2_test --suite core
    if ($LASTEXITCODE -eq 0) { python experiments/analyze.py --model deit_small --split v2_test --suite distill }
    if ($LASTEXITCODE -eq 0) { python experiments/analyze.py --model vit_base --split v2_test --suite core }
}
Log "=== queue complete"
