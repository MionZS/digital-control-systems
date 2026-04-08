# Run Box4 identification and validation pipeline
Set-StrictMode -Version Latest

$root = Resolve-Path "${PSScriptRoot}/.." | Select-Object -ExpandProperty Path
Push-Location $root

Write-Host "Running identification on EstimacaoBox4..."
uv run python Scripts/automatic_order_identification.py --control input/EstimacaoBox4/controle.lvm --output input/EstimacaoBox4/saida.lvm --out-json output/box4/identified_model.json --out-plot output/box4/identified_model_plot.png --time-scale 0.1

Write-Host "Validating identified model on ValidacaoBox4..."
uv run python Scripts/validate_identified_model.py --model-json output/box4/identified_model.json --control input/ValidacaoBox4/controle.lvm --output input/ValidacaoBox4/saida.lvm --out-json output/box4/validation_results_auto_selected.json --out-plot output/box4/validation_plot_auto_selected.png --time-scale 0.1

Pop-Location
Write-Host "Box4 pipeline finished. Results in output/box4."