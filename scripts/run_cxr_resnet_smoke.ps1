# ResNet-50 (resnet50-res512-all) smoke pipeline for SIIM pneumothorax.
# Phase 1.7-seam: uses load_classifier() ResNet branch.
#
# Pre-flight checks (do once on a fresh clone):
#   wsl.exe --cd /mnt/c/Users/Dmytro.Valantsevych/Downloads/master_thesis_draft_explainAI `
#       python3 -m pip install -e . --no-deps --no-build-isolation
#
# Notes:
# - ResNet-50 native input is 512x512; do NOT lower --image-size or you double-
#   downsample. Faithfulness, Grad-CAM target_layer, and v2 calibration must all
#   match this resolution.
# - The v1 DenseNet calibration top-fractions are stale for ResNet because the
#   signed attribution distribution differs. We regenerate a ResNet-specific v2
#   calibration first, then point the smoke run at it.
# - All outputs land under a fresh iter_XX folder per AGENTS.md housekeeping.
# - Estimated wall time at 512x512 with --max-positive 6 and 8/8 IG/GradientSHAP
#   samples + coarse occlusion: ~15-25 min on CPU, ~3-6 min on CUDA. For the
#   full calibration (--max-positive 200) budget ~60-90 min on CUDA; if the run
#   exceeds 30 min on your machine, execute it manually instead of via agent.

$ErrorActionPreference = "Stop"
$ProjectRoot = "/mnt/c/Users/Dmytro.Valantsevych/Downloads/master_thesis_draft_explainAI"
$Weights = "resnet50-res512-all"
$ImageSize = 512

$CalibDir = "outputs/iter_30_calibration_v2_resnet50"
$SmokeDir = "outputs/iter_31_resnet50_smoke"

Write-Host "=== Step 1/2: regenerate v2 calibration for $Weights ==="
wsl.exe --cd $ProjectRoot python3 scripts/calibrate_cxr_xai_thresholds.py `
    --weights $Weights `
    --image-size $ImageSize `
    --device auto `
    --split train `
    --max-positive 200 `
    --random-sample `
    --seed 20260518 `
    --ig-steps 8 `
    --gradshap-samples 8 `
    --occlusion-patch-size 64 `
    --occlusion-stride 32 `
    --output-dir $CalibDir

$CalibCsv = "$CalibDir/calibrated_thresholds_v2.csv"
Write-Host ""
Write-Host "=== Step 2/2: ResNet-50 smoke run (positive cases) ==="
wsl.exe --cd $ProjectRoot python3 scripts/run_cxr_torchxray_smoke.py `
    --weights $Weights `
    --image-size $ImageSize `
    --device auto `
    --split test `
    --max-positive 6 `
    --random-sample `
    --seed 20260518 `
    --ig-steps 8 `
    --gradshap-samples 8 `
    --occlusion-patch-size 64 `
    --occlusion-stride 32 `
    --faithfulness-baseline black `
    --calibrated-fractions $CalibCsv `
    --classifier-threshold 0.62 `
    --output-dir $SmokeDir

Write-Host ""
Write-Host "Done. Inspect:"
Write-Host "  $CalibDir/calibrated_thresholds_v2.csv"
Write-Host "  $SmokeDir/case_metrics.csv"
Write-Host "  $SmokeDir/method_summary.csv"
Write-Host ""
Write-Host "Reminder: classifier threshold 0.62 was calibrated for DenseNet-121-all"
Write-Host "on SIIM train. ResNet-50 may need its own threshold sweep "
Write-Host "(scripts/evaluate_cxr_torchxray_model.py) before final reporting."
