# Run motor identification pipeline (degrausMotorTQ)
Set-StrictMode -Version Latest

$root = Resolve-Path "${PSScriptRoot}/.." | Select-Object -ExpandProperty Path
Push-Location $root

Write-Host "Running identification on degrausMotorTQ..."
uv run python Scripts/automatic_order_identification.py --control input/degrausMotorTQ/controle.lvm --output input/degrausMotorTQ/saida.lvm --out-json output/automatic_order_motor/identified_model.json --out-plot output/automatic_order_motor/identified_model_plot.png --time-scale 0.1

Pop-Location
Write-Host "Motor pipeline finished. Results in output/automatic_order_motor."