$ErrorActionPreference = 'Stop'
$projectRoot = $PSScriptRoot
$python = Join-Path $projectRoot '.venv\Scripts\python.exe'
Set-Location $projectRoot

& $python (Join-Path $projectRoot 'prepare_forest_fire_dataset.py') --overwrite
if ($LASTEXITCODE -ne 0) { throw "Dataset preparation failed with exit code $LASTEXITCODE." }

& $python (Join-Path $projectRoot 'train_forest_fire_detector.py') --epochs 40 --batch 8 --device 0
if ($LASTEXITCODE -ne 0) { throw "Training failed with exit code $LASTEXITCODE." }

& $python (Join-Path $projectRoot 'evaluate_forest_fire_detector.py') --device 0
if ($LASTEXITCODE -ne 0) { throw "Evaluation failed with exit code $LASTEXITCODE." }
