# Run closed-loop thermal simulation using identified impulse model
Set-StrictMode -Version Latest

$root = Resolve-Path "${PSScriptRoot}/.." | Select-Object -ExpandProperty Path
Push-Location $root

$outDir = "output/impulse_response/closed_loop"
New-Item -ItemType Directory -Force -Path $outDir | Out-Null

uv run python Scripts/simulate_impulse_closed_loop.py `
    --model-json output/impulse_response/degrau1_lampada_26032026/identified_model.json `
    --disturbance-csv output/impulse_response/degrau1_lampada_26032026/degrau1_lampada_26032026.csv `
    --out-json output/impulse_response/closed_loop/closed_loop_results.json `
    --out-plot output/impulse_response/closed_loop/closed_loop_plot.png `
    --delta-ref 5.0

Pop-Location
Write-Host "Closed-loop simulation finished. See output/impulse_response/closed_loop/."