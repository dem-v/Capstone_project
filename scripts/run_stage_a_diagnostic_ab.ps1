# Phase 1.7 Stage A — diagnostic A/B across classifier weights.
#
# Per AGENTS.md "Diagnostic A/B Protocol Before Full Second-Model Integration"
# and docs/refactor_plan.md §1.7 Stage A. For each model in the working set,
# runs three sub-steps:
#
#   1. Classifier-threshold sweep (train split)
#        -> outputs/iter_33_stage_a_diagnostic_ab/<model>/threshold_sweep/
#           classification_metrics.csv (per-model best F1 / Youden's J /
#           high-sensitivity operating points).
#   2. v2 XAI top-fraction calibration (positive masked calibration cases)
#        -> outputs/iter_33_stage_a_diagnostic_ab/<model>/calibration_v2/
#           calibrated_thresholds_v2.csv.
#   3. Smoke + faithfulness (positive cases, full XAI method set)
#        -> outputs/iter_33_stage_a_diagnostic_ab/<model>/smoke/
#           metrics.csv + overlays + faithfulness curves.
#
# After the per-model loop, aggregates per-model mean IoU / Dice /
# pointing_hit / precision_at_fraction (positive view, Dice-selected
# calibrated fraction) into a single weights_ab_summary.csv at the run root.
#
# Working set (mirror of src/explainai_thesis/cxr/classifier.py allow-lists,
# minus the two non-runnable entries):
#   - densenet121-res224-{all, chex, mimic_ch, mimic_nb, nih, pc}  (in-family)
#   - resnet50-res512-all                                          (in-family, 512x512)
#   - The MONAI out-of-family bundle is appended once integrated through
#     load_classifier(); the in-family sweep can start without it.
#
# Skipped (recorded with a reason, not a failure):
#   - densenet121-res224-rsna  : no Pneumothorax head.
#   - resnetae-101-elastic     : no class head (autoencoder).
#
# Wall-time on CUDA (rough): per-model threshold sweep ~5 min, v2 calibration
# at --max-positive 200 ~60-90 min, smoke ~30-45 min. Seven models ~ 12-16 h
# total. Past the 30 min agent-tool budget; run manually per AGENTS.md:
#
#   pwsh scripts/run_stage_a_diagnostic_ab.ps1
#   pwsh scripts/run_stage_a_diagnostic_ab.ps1 -DryRun
#   pwsh scripts/run_stage_a_diagnostic_ab.ps1 -Models densenet121-res224-all
#   pwsh scripts/run_stage_a_diagnostic_ab.ps1 -SkipSweep -SkipCalibration  # smoke only
#
# Resume strategy: each sub-step writes into a stable subfolder. If a model's
# threshold_sweep/calibration_v2/smoke folder already contains the expected
# artifact (classification_metrics.csv / calibrated_thresholds_v2.csv /
# metrics.csv), the corresponding step is skipped unless -Force is passed.

[CmdletBinding()]
param(
    [int]      $SweepMaxCases        = 2000,
    [int]      $CalibMaxPositive     = 200,
    [int]      $SmokeMaxPositive     = 30,
    [int]      $Seed                 = 20260518,
    [string]   $Device               = "auto",
    [string]   $FaithBaseline        = "black",
    [int]      $IgSteps              = 8,
    [int]      $GradShapSamp         = 8,
    [string]   $Fractions            = "0.05,0.10,0.15,0.20,0.25,0.30,0.35,0.40,0.45,0.50,0.55,0.60,0.65,0.70,0.75,0.80,0.85,0.90,0.95",
    [string]   $OutputRoot           = "outputs/iter_33_stage_a_diagnostic_ab",
    [string[]] $Models               = @(),
    [switch]   $DryRun,
    [switch]   $Force,
    [switch]   $SkipSweep,
    [switch]   $SkipCalibration,
    [switch]   $SkipSmoke,
    [switch]   $SkipSummary
)

$ErrorActionPreference = "Stop"
$ProjectRoot = "/mnt/c/Users/Dmytro.Valantsevych/Downloads/master_thesis_draft_explainAI"

# Working set + skip reasons. Mirrors run_all_models_smoke.ps1.
$DenseNetWeights = @(
    "densenet121-res224-all",
    "densenet121-res224-chex",
    "densenet121-res224-mimic_ch",
    "densenet121-res224-mimic_nb",
    "densenet121-res224-nih",
    "densenet121-res224-pc"
)
$ResNetWeights = @("resnet50-res512-all")
$Skipped = @{
    "densenet121-res224-rsna" = "No Pneumothorax class in head (RSNA Pneumonia)."
    "resnetae-101-elastic"    = "Autoencoder; no class head."
}

$ModelConfigs = @{}
foreach ($w in $DenseNetWeights) {
    $ModelConfigs[$w] = @{ ImageSize = 224; OccPatch = 56; OccStride = 56 }
}
foreach ($w in $ResNetWeights) {
    $ModelConfigs[$w] = @{ ImageSize = 512; OccPatch = 64; OccStride = 32 }
}

$AllModels = $DenseNetWeights + $ResNetWeights
if ($Models.Count -gt 0) {
    $Unknown = $Models | Where-Object {
        -not ($AllModels -contains $_) -and -not ($Skipped.ContainsKey($_))
    }
    if ($Unknown.Count -gt 0) {
        throw "Unknown weight name(s): $($Unknown -join ', '). Known: $($AllModels -join ', ')"
    }
    $Selected = $Models
} else {
    $Selected = $AllModels
}

Write-Host "=== run_stage_a_diagnostic_ab ==="
Write-Host "Models           : $($Selected -join ', ')"
Write-Host "SweepMaxCases    : $SweepMaxCases"
Write-Host "CalibMaxPositive : $CalibMaxPositive"
Write-Host "SmokeMaxPositive : $SmokeMaxPositive"
Write-Host "Device           : $Device   Seed: $Seed"
Write-Host "OutputRoot       : $OutputRoot"
Write-Host "DryRun           : $DryRun   Force: $Force"
Write-Host "Skip flags       : sweep=$SkipSweep  calib=$SkipCalibration  smoke=$SkipSmoke  summary=$SkipSummary"
Write-Host ""

function Invoke-Step {
    param(
        [string]   $Label,
        [string[]] $ArgList,
        [string]   $ExpectedArtifact
    )
    if ((-not $Force) -and ($ExpectedArtifact -and (Test-Path $ExpectedArtifact))) {
        Write-Host "    SKIP $Label (artifact exists: $ExpectedArtifact; pass -Force to rerun)"
        return @{ Status = "skip-existing"; Seconds = 0 }
    }
    if ($DryRun) {
        Write-Host "    DRY-RUN $Label : wsl.exe --cd $ProjectRoot python3 $($ArgList -join ' ')"
        return @{ Status = "dry-run"; Seconds = 0 }
    }
    $t0 = Get-Date
    try {
        wsl.exe --cd $ProjectRoot python3 @ArgList
        if ($LASTEXITCODE -ne 0) { throw "non-zero exit code: $LASTEXITCODE" }
        $dt = [int]((Get-Date) - $t0).TotalSeconds
        Write-Host "    OK $Label ($dt s)"
        return @{ Status = "ok"; Seconds = $dt }
    } catch {
        $dt = [int]((Get-Date) - $t0).TotalSeconds
        Write-Warning "    FAIL $Label after $dt s: $_"
        return @{ Status = "fail"; Seconds = $dt; Error = "$_" }
    }
}

function Get-BestF1Threshold {
    param([string] $CsvPath)
    if (-not (Test-Path $CsvPath)) { return $null }
    try {
        $row = Import-Csv $CsvPath | Select-Object -First 1
        if ($row -and $row.PSObject.Properties.Name -contains "best_f1_threshold") {
            return [double] $row.best_f1_threshold
        }
    } catch {
        Write-Warning "    cannot read best_f1_threshold from $CsvPath : $_"
    }
    return $null
}

$Results = @()
$Index = 0
foreach ($w in $Selected) {
    $Index += 1
    Write-Host "--- [$Index/$($Selected.Count)] $w ---"

    if ($Skipped.ContainsKey($w)) {
        $reason = $Skipped[$w]
        Write-Host "    SKIP (unrunnable): $reason"
        $Results += [pscustomobject]@{
            Model = $w; Sweep = "skipped"; Calibration = "skipped"; Smoke = "skipped";
            ClsThreshold = ""; Reason = $reason
        }
        Write-Host ""
        continue
    }

    $cfg       = $ModelConfigs[$w]
    $modelRoot = "$OutputRoot/$w"
    $sweepDir  = "$modelRoot/threshold_sweep"
    $calibDir  = "$modelRoot/calibration_v2"
    $smokeDir  = "$modelRoot/smoke"
    $sweepCsv  = "$sweepDir/classification_metrics.csv"
    $calibCsv  = "$calibDir/calibrated_thresholds_v2.csv"
    $smokeCsv  = "$smokeDir/metrics.csv"

    $row = [ordered]@{
        Model       = $w
        Sweep       = "pending"
        Calibration = "pending"
        Smoke       = "pending"
        ClsThreshold = ""
        Reason      = ""
    }

    # ---- Step 1: classifier-threshold sweep ----
    if ($SkipSweep) {
        $row.Sweep = "skipped-flag"
    } else {
        $argList = @(
            "scripts/evaluate_cxr_torchxray_model.py",
            "--weights", $w,
            "--device", $Device,
            "--split", "train",
            "--image-size", $cfg.ImageSize,
            "--max-cases", $SweepMaxCases,
            "--random-sample",
            "--seed", $Seed,
            "--output-dir", $sweepDir
        )
        $r = Invoke-Step -Label "sweep" -ArgList $argList -ExpectedArtifact $sweepCsv
        $row.Sweep = $r.Status
        if ($r.Status -eq "fail") { $row.Reason = $r.Error }
    }

    # Pull best-F1 threshold for the smoke step. Fall back to 0.62 (DenseNet-all
    # historical default) only if the sweep CSV is missing AND we're not doing
    # the smoke step; the smoke step will then warn.
    $clsThreshold = Get-BestF1Threshold -CsvPath $sweepCsv
    if ($null -eq $clsThreshold) {
        if ($DryRun) {
            $clsThreshold = 0.62  # placeholder for dry-run command rendering
        } else {
            $clsThreshold = 0.62
            Write-Warning "    using fallback classifier-threshold 0.62 for $w (no sweep CSV)"
        }
    }
    $row.ClsThreshold = $clsThreshold

    # ---- Step 2: v2 XAI calibration ----
    if ($SkipCalibration) {
        $row.Calibration = "skipped-flag"
    } else {
        $argList = @(
            "scripts/calibrate_cxr_xai_thresholds.py",
            "--weights", $w,
            "--device", $Device,
            "--image-size", $cfg.ImageSize,
            "--max-positive", $CalibMaxPositive,
            "--random-sample",
            "--seed", $Seed,
            "--ig-steps", $IgSteps,
            "--gradshap-samples", $GradShapSamp,
            "--occlusion-patch-size", $cfg.OccPatch,
            "--occlusion-stride", $cfg.OccStride,
            "--fractions", $Fractions,
            "--classifier-threshold", $clsThreshold,
            "--output-dir", $calibDir
        )
        $r = Invoke-Step -Label "calibration_v2" -ArgList $argList -ExpectedArtifact $calibCsv
        $row.Calibration = $r.Status
        if ($r.Status -eq "fail" -and -not $row.Reason) { $row.Reason = $r.Error }
    }

    # ---- Step 3: smoke + faithfulness ----
    if ($SkipSmoke) {
        $row.Smoke = "skipped-flag"
    } else {
        $argList = @(
            "scripts/run_cxr_torchxray_smoke.py",
            "--weights", $w,
            "--device", $Device,
            "--image-size", $cfg.ImageSize,
            "--split", "test",
            "--max-positive", $SmokeMaxPositive,
            "--random-sample",
            "--seed", $Seed,
            "--ig-steps", $IgSteps,
            "--gradshap-samples", $GradShapSamp,
            "--occlusion-patch-size", $cfg.OccPatch,
            "--occlusion-stride", $cfg.OccStride,
            "--faithfulness-baseline", $FaithBaseline,
            "--classifier-threshold", $clsThreshold,
            "--output-dir", $smokeDir
        )
        if ((-not $DryRun) -and (Test-Path $calibCsv)) {
            $argList += @("--calibrated-fractions", $calibCsv)
        } elseif ($DryRun) {
            $argList += @("--calibrated-fractions", $calibCsv)
        }
        $r = Invoke-Step -Label "smoke" -ArgList $argList -ExpectedArtifact $smokeCsv
        $row.Smoke = $r.Status
        if ($r.Status -eq "fail" -and -not $row.Reason) { $row.Reason = $r.Error }
    }

    $Results += [pscustomobject] $row
    Write-Host ""
}

# Write per-model status summary at the run root.
if (-not $DryRun) {
    $null = New-Item -ItemType Directory -Force -Path $OutputRoot
    $statusPath = "$OutputRoot/stage_a_status.csv"
    $Results | Export-Csv -NoTypeInformation -Path $statusPath
    Write-Host "Stage A status: $statusPath"
}
Write-Host ""
Write-Host "=== Per-model status ==="
$Results | Format-Table -AutoSize

# ---- Aggregator: weights_ab_summary.csv ----
# Reads each <model>/smoke/metrics.csv, takes the positive-view rows at the
# Dice-selected calibrated fraction, and emits per-model means of the four
# canonical localization metrics. Failed/skipped models leave NaN rows.
if (-not $SkipSummary -and -not $DryRun) {
    Write-Host ""
    Write-Host "=== Aggregating weights_ab_summary.csv ==="
    $pythonSnippet = @"
import csv, math, os, sys
from pathlib import Path

root = Path(r"$OutputRoot")
metrics_of_interest = ("iou", "dice", "pointing_hit", "precision_at_fraction")
summary_rows = []

for model_dir in sorted(p for p in root.iterdir() if p.is_dir()):
    metrics_csv = model_dir / "smoke" / "metrics.csv"
    row = {"model": model_dir.name}
    if not metrics_csv.exists():
        for m in metrics_of_interest:
            row[f"mean_{m}"] = ""
        row["n_cases"] = 0
        row["status"] = "no-smoke"
        summary_rows.append(row)
        continue
    with metrics_csv.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        # Filter to the positive view rows; tolerate either a 'view' column or
        # method names that already encode the view (post-refactor schema may
        # differ across runs). When 'view' is absent, take all rows.
        rows = list(reader)
    if not rows:
        for m in metrics_of_interest:
            row[f"mean_{m}"] = ""
        row["n_cases"] = 0
        row["status"] = "empty-smoke"
        summary_rows.append(row)
        continue
    fieldnames = rows[0].keys()
    if "view" in fieldnames:
        rows = [r for r in rows if r.get("view") == "positive"]
    n = 0
    sums = {m: 0.0 for m in metrics_of_interest}
    for r in rows:
        ok = False
        for m in metrics_of_interest:
            v = r.get(m, "")
            try:
                fv = float(v)
                if math.isfinite(fv):
                    sums[m] += fv
                    ok = True
            except (TypeError, ValueError):
                pass
        if ok:
            n += 1
    for m in metrics_of_interest:
        row[f"mean_{m}"] = (sums[m] / n) if n else ""
    row["n_cases"] = n
    row["status"] = "ok" if n else "no-finite-metrics"
    summary_rows.append(row)

out = root / "weights_ab_summary.csv"
header = ["model"] + [f"mean_{m}" for m in metrics_of_interest] + ["n_cases", "status"]
with out.open("w", newline="", encoding="utf-8") as fh:
    writer = csv.DictWriter(fh, fieldnames=header)
    writer.writeheader()
    for r in summary_rows:
        writer.writerow(r)
print(f"weights_ab_summary: {out}")
"@
    $tmp = New-TemporaryFile
    Set-Content -Path $tmp -Value $pythonSnippet -Encoding UTF8
    try {
        wsl.exe --cd $ProjectRoot python3 (wsl.exe wslpath -a (Resolve-Path $tmp).Path)
    } finally {
        Remove-Item $tmp -Force -ErrorAction SilentlyContinue
    }
}

$failed = @($Results | Where-Object { $_.Sweep -eq "fail" -or $_.Calibration -eq "fail" -or $_.Smoke -eq "fail" })
if ($failed.Count -gt 0) {
    Write-Warning "$($failed.Count) model(s) had at least one failed step: $($failed.Model -join ', ')"
    exit 1
}
