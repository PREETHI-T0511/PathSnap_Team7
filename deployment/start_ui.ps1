$ErrorActionPreference = "Stop"
$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$uiPath = Join-Path $projectRoot "ui"
Set-Location -LiteralPath $uiPath
Write-Host "UI listening on port 8080. Open http://<this-laptop-ip>:8080 and enter the API laptop address in the page."
python -m http.server 8080 --bind 0.0.0.0
