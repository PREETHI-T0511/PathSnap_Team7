$ErrorActionPreference = "Stop"
$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$modelPath = Join-Path $projectRoot "artifacts\tuned_best_model.joblib"
$evaluatorPath = Join-Path $projectRoot "data\evaluator_test"
if (-not (Test-Path -LiteralPath $modelPath -PathType Leaf)) {
    throw "Model artifact not found: $modelPath"
}
if (-not (Test-Path -LiteralPath $evaluatorPath -PathType Container)) {
    throw "Create data\evaluator_test and copy the reserved evaluator trips there on this API laptop. Do not expose its .route files to the UI laptop."
}
$env:PATHSNAP_MODEL_PATH = $modelPath
$env:PATHSNAP_EVALUATOR_DATA_DIR = $evaluatorPath
if (-not $env:PATHSNAP_CORS_ORIGINS) { $env:PATHSNAP_CORS_ORIGINS = "*" }
Set-Location -LiteralPath $projectRoot
python -m uvicorn api.main:app --host 0.0.0.0 --port 8000
