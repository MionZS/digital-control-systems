# Run impulse response pipeline: convert TXT -> CSV/LVM, then identify model for each
Set-StrictMode -Version Latest

$root = Resolve-Path "${PSScriptRoot}/.." | Select-Object -ExpandProperty Path
Push-Location $root

Write-Host "Converting TXT files to CSV and LVMs..."
uv run python Scripts/convert_impulse_txt_to_csv.py

Get-ChildItem -Path output/impulse_response -Directory | ForEach-Object {
    $folder = $_.FullName
    Write-Host "Identifying model for folder: $folder"
    $control = Join-Path $folder 'control.lvm'
    $output = Join-Path $folder 'output.lvm'
    $outjson = Join-Path $folder 'identified_model.json'
    $outplot = Join-Path $folder 'identified_model_plot.png'
    uv run python Scripts/automatic_order_identification.py --control $control --output $output --out-json $outjson --out-plot $outplot --time-scale 1.0
}

Pop-Location
Write-Host "Impulse pipeline finished. Results in output/impulse_response."