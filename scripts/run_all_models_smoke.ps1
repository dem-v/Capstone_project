# Smoke-test all supported classifier weights behind the load_classifier() seam.
#
# Iterates over every entry in:
#   - TORCHXRAYVISION_DENSENET_WEIGHTS    (7 weights, 224x224 native)
#   - TORCHXRAYVISION_RESNET_WEIGHTS      (1 weight,  512x512 native)
#   - TORCHXRAYVISION_AUTOENCODER_WEIGHTS (1 weight,  ResNetAE, 224x224 native)
#
# ResNetAE handling: ResNetAE has no Pneumothorax head and the current
# pneumothorax-target XAI pipeline cannot attribute against an undefined
# class score. `load_classifier` enforces this by rejecting the default
# `pathology='Pneumothorax'` call for ResNetAE. Per user request, ResNetAE
# is still listed in this sweep so the orchestrator is the single source of
# truth for "every supported weight", but it is recorded as `skipped`
# (with a clear reason) rather than executed. Re-enabling ResNetAE here will
# require a separate reconstruction / latent-space evaluation script; see
# AGENTS.md "Modality Coverage" for the planned scope.
#
# This is a SMOKE pass, not a thesis-grade evaluation:
# - Few positive cases per model (--max-positive).
# - Exploratory IG/GradientSHAP/Occlusion settings (8/8, coarse occlusion).
# - The DenseNet v1 (pre-refactor) calibrated thresholds are NOT used; smoke
#   runs without --calibrated-fractions to keep this script self-contained
#   across all 8 models. Per-model v2 calibrations are out of scope here.
# - Classifier threshold reuses the DenseNet-calibrated 0.62; this is fine for
#   a smoke pass but must be re-derived per model before any held-out report.
#
# Outputs land under one shared parent folder, one subfolder per model:
#   outputs/iter_32_all_models_smoke/<weights_name>/
# This keeps comparison easy and respects AGENTS.md output-folder rules.
#
# Usage:
#   pwsh scripts/run_all_models_smoke.ps1
#   pwsh scripts/run_all_models_smoke.ps1 -MaxPositive 4 -Models densenet121-res224-all,resnet50-res512-all
#   pwsh scripts/run_all_models_smoke.ps1 -DryRun
#
# Estimated wall time per model (--max-positive 6, --device auto on CUDA):
#   DenseNet-121 @ 224x224 : ~2-4 min
#   ResNet-50    @ 512x512 : ~5-10 min
# Full 8-model sweep on CUDA: ~30-60 min. On CPU expect 4-8x longer; if your
# machine pushes total runtime beyond ~30 min, run this manually (AGENTS.md).

[CmdletBinding()]
param(
    [int]    $MaxPositive   = 6,
    [int]    $Seed          = 20260518,
    [string] $Split         = "test",
    [string] $Device        = "auto",
    [string] $FaithBaseline = "black",
    [double] $ClsThreshold  = 0.62,
    [int]    $IgSteps       = 8,
    [int]    $GradShapSamp  = 8,
    [string] $OutputRoot    = "outputs/iter_32_all_models_smoke",
    [string[]] $Models      = @(),  # empty = all
    [switch] $DryRun
)

$ErrorActionPreference = "Stop"
$ProjectRoot = "/mnt/c/Users/Dmytro.Valantsevych/Downloads/master_thesis_draft_explainAI"

# Mirror of src/explainai_thesis/cxr/classifier.py allow-lists (all three
# families). If you add weights to the seam, add them here too. ResNetAE is
# listed but auto-skipped in the loop below; see header comment.
$DenseNetWeights = @(
    "densenet121-res224-all",
    "densenet121-res224-chex",
    "densenet121-res224-mimic_ch",
    "densenet121-res224-mimic_nb",
    "densenet121-res224-rsna",
    "densenet121-res224-nih",
    "densenet121-res224-pc"
)
# Subset of DenseNet weights whose classification head does NOT contain a
# Pneumothorax class (verified at runtime: `densenet121-res224-rsna` was
# trained on the RSNA Pneumonia challenge and its `pathologies` list only
# contains 'Pneumonia' / 'Lung Opacity'). These cannot be smoke-tested
# through the current pneumothorax-target XAI pipeline; the loop skips
# them explicitly with a clear reason rather than failing.
$NoPneumothoraxWeights = @(
    "densenet121-res224-rsna"
)
$ResNetWeights = @(
    "resnet50-res512-all"
)
# Autoencoders are exposed via load_classifier() but are NOT pneumothorax
# classifiers; the current XAI pipeline does not support them. Listed here
# only so the sweep summary reflects every weight in the seam allow-lists.
$AutoencoderWeights = @(
    "resnetae-101-elastic"
)

# Per-model native input + occlusion patch/stride (scaled with image size so
# the occlusion grid covers a comparable fraction of the image across models).
$ModelConfigs = @{}
foreach ($w in $DenseNetWeights) {
    $ModelConfigs[$w] = @{ ImageSize = 224; OccPatch = 56; OccStride = 56 }
}
foreach ($w in $ResNetWeights) {
    $ModelConfigs[$w] = @{ ImageSize = 512; OccPatch = 64; OccStride = 32 }
}
# Config kept for completeness; not actually used because autoencoders are
# skipped below before any smoke command is built.
foreach ($w in $AutoencoderWeights) {
    $ModelConfigs[$w] = @{ ImageSize = 224; OccPatch = 56; OccStride = 56 }
}

$AllModels = $DenseNetWeights + $ResNetWeights + $AutoencoderWeights
if ($Models.Count -gt 0) {
    $Selected = $Models | Where-Object { $AllModels -contains $_ }
    $Unknown  = $Models | Where-Object { -not ($AllModels -contains $_) }
    if ($Unknown.Count -gt 0) {
        throw "Unknown weight name(s): $($Unknown -join ', '). Known: $($AllModels -join ', ')"
    }
} else {
    $Selected = $AllModels
}

Write-Host "=== run_all_models_smoke ==="
Write-Host "Models       : $($Selected -join ', ')"
Write-Host "MaxPositive  : $MaxPositive   Split: $Split   Seed: $Seed   Device: $Device"
Write-Host "OutputRoot   : $OutputRoot"
Write-Host "DryRun       : $DryRun"
Write-Host ""

$Results = @()
$Index = 0
foreach ($w in $Selected) {
    $Index += 1
    $cfg = $ModelConfigs[$w]
    $outDir = "$OutputRoot/$w"
    Write-Host "--- [$Index/$($Selected.Count)] $w  (image_size=$($cfg.ImageSize)) ---"
    Write-Host "    out: $outDir"

    # Autoencoders have no Pneumothorax head -> incompatible with the current
    # pneumothorax-target XAI pipeline. Skip explicitly (not a failure).
    if ($AutoencoderWeights -contains $w) {
        $skipReason = "ResNetAE has no Pneumothorax head; current XAI pipeline requires a class score. Re-enable via a dedicated reconstruction/latent script."
        Write-Host "    SKIP: $skipReason"
        $Results += [pscustomobject]@{ Model = $w; Status = "skipped"; Seconds = 0; OutputDir = $outDir; Error = $skipReason }
        Write-Host ""
        continue
    }

    # Some classifier heads do not include 'Pneumothorax' (e.g. the RSNA
    # Pneumonia-challenge weights expose only 'Pneumonia' / 'Lung Opacity').
    # load_classifier() would raise; skip with a clear reason instead.
    if ($NoPneumothoraxWeights -contains $w) {
        $skipReason = "Classifier head has no 'Pneumothorax' class (trained on a different target set); cannot be evaluated through the pneumothorax-target XAI pipeline."
        Write-Host "    SKIP: $skipReason"
        $Results += [pscustomobject]@{ Model = $w; Status = "skipped"; Seconds = 0; OutputDir = $outDir; Error = $skipReason }
        Write-Host ""
        continue
    }

    $argList = @(
        "scripts/run_cxr_torchxray_smoke.py",
        "--weights", $w,
        "--image-size", $cfg.ImageSize,
        "--device", $Device,
        "--split", $Split,
        "--max-positive", $MaxPositive,
        "--random-sample",
        "--seed", $Seed,
        "--ig-steps", $IgSteps,
        "--gradshap-samples", $GradShapSamp,
        "--occlusion-patch-size", $cfg.OccPatch,
        "--occlusion-stride", $cfg.OccStride,
        "--faithfulness-baseline", $FaithBaseline,
        "--classifier-threshold", $ClsThreshold,
        "--output-dir", $outDir
    )

    if ($DryRun) {
        Write-Host "    DRY-RUN: wsl.exe --cd $ProjectRoot python3 $($argList -join ' ')"
        $Results += [pscustomobject]@{ Model = $w; Status = "dry-run"; OutputDir = $outDir }
        continue
    }

    $t0 = Get-Date
    try {
        wsl.exe --cd $ProjectRoot python3 @argList
        if ($LASTEXITCODE -ne 0) {
            throw "non-zero exit code: $LASTEXITCODE"
        }
        $dt = [int]((Get-Date) - $t0).TotalSeconds
        Write-Host "    OK  ($dt s)"
        $Results += [pscustomobject]@{ Model = $w; Status = "ok"; Seconds = $dt; OutputDir = $outDir }
    } catch {
        $dt = [int]((Get-Date) - $t0).TotalSeconds
        Write-Warning "    FAIL after $dt s: $_"
        $Results += [pscustomobject]@{ Model = $w; Status = "fail"; Seconds = $dt; OutputDir = $outDir; Error = "$_" }
        # Continue the sweep rather than abort; one broken weight should not
        # block the remaining smoke checks. Failures are visible in summary.csv.
    }
    Write-Host ""
}

# Write a flat sweep summary at the OutputRoot so the next agent / reviewer can
# see at a glance which weights smoke-passed and how long each took.
if (-not $DryRun) {
    $summaryPath = "$OutputRoot/sweep_summary.csv"
    $null = New-Item -ItemType Directory -Force -Path $OutputRoot
    $Results | Export-Csv -NoTypeInformation -Path $summaryPath
    Write-Host "Sweep summary: $summaryPath"
}

Write-Host ""
Write-Host "=== Summary ==="
$Results | Format-Table -AutoSize

$failed = @($Results | Where-Object { $_.Status -eq "fail" })
if ($failed.Count -gt 0) {
    Write-Warning "$($failed.Count) model(s) failed: $($failed.Model -join ', ')"
    exit 1
}
