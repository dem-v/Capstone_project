# Builds the static review workbook from the completed 40-case ResNet diagnostic render.
# Run from any directory in PowerShell.

$ErrorActionPreference = 'Stop'

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$repoRoot = Resolve-Path (Join-Path $scriptDir '..')
Set-Location $repoRoot

$selectionCsv = 'outputs/iter_46_resnet_review_candidate_selection_balanced40/selected_manual_review_cases.csv'
$diagnosticsDir = 'outputs/iter_47_resnet_review_diagnostics_balanced40_smoothed_faithfulness'
$outputDir = 'outputs/iter_48_resnet_review_workbook_balanced40_smoothed_faithfulness/review'

if (-not (Test-Path $selectionCsv)) {
    throw "Missing selection CSV: $selectionCsv"
}
if (-not (Test-Path $diagnosticsDir)) {
    throw "Missing diagnostics render directory: $diagnosticsDir"
}

Write-Host 'Building 40-case ResNet review workbook...'
Write-Host "Selection CSV: $selectionCsv"
Write-Host "Diagnostics:   $diagnosticsDir"
Write-Host "Output:        $outputDir"

wsl.exe python3 scripts/build_review_workbook.py `
    --selection-csv $selectionCsv `
    --diagnostics-dir $diagnosticsDir `
    --output-dir $outputDir

if ($LASTEXITCODE -ne 0) {
    throw "Workbook generation failed with exit code $LASTEXITCODE"
}

$requiredOutputs = @(
    (Join-Path $outputDir 'index.html'),
    (Join-Path $outputDir 'scores_template.csv'),
    (Join-Path $outputDir 'INSTRUCTIONS.md')
)
foreach ($path in $requiredOutputs) {
    if (-not (Test-Path $path)) {
        throw "Expected workbook output was not created: $path"
    }
}

Write-Host 'Workbook ready.'
Write-Host "Open: $outputDir/index.html"