# Launches one named experiment detached, logs under generalize\runs\<job>\.
#   .\run_jobs.ps1 -Job int-d192-split -Threads 10
#   .\run_jobs.ps1 -Job float-ceiling-d384-long -Device cuda
#   .\run_jobs.ps1 -List
# Every path is relative to this file, so the checkout can live anywhere.
param(
    [string]$Job = "",
    [int]$Threads = 8,
    [string]$Device = "cpu",
    [string]$Python = "python",
    [int]$Epochs = 0,
    [switch]$List
)
$ErrorActionPreference = "Stop"
$here = $PSScriptRoot
$root = Split-Path $here -Parent
$runs = Join-Path $here "runs"
New-Item -ItemType Directory -Force $runs | Out-Null
$env:PYTHONIOENCODING = "utf-8"; $env:PYTHONUTF8 = "1"; $env:PYTHONUNBUFFERED = "1"

# the integer trainer's full-corpus short-sentence configuration (66,853 / 3,518 pairs)
$intCommon = @("-X", "utf8", "-u", "--mode", "train", "--corpus", "both", "--max-pairs", "0", "--base-pairs", "0",
    "--vocab-size", "5000", "--min-char-count", "20", "--max-len", "16", "--max-tok", "14", "--tok-mode", "char",
    "--valid-frac", "0.05", "--k-enc", "2", "--k-dec", "4", "--dec-pattern", "ccaf", "--lr-bits", "1",
    "--hidden-bits", "16", "--state-bits", "16", "--weight-bits", "8", "--accum-bits", "32", "--positional", "16",
    "--score-exp2", "3", "--batch", "32", "--eval-every", "1", "--patience", "8", "--max-eval", "512", "--seed", "0",
    "--show-samples", "10", "--snapshot-epochs", "0", "1", "2", "4", "8", "16")

$jobs = [ordered]@{
    # integer BlockSGD, split momentum (target side 7, source pathway 0), d192
    "int-d192-split" = @{ script = (Join-Path $root "jako_train_split.py"); epochs = 30
        args = $intCommon + @("--dim", "192", "--ffn-hidden", "384", "--momentum-bits", "7", "--source-momentum-bits", "0") }
    # integer BlockSGD, uniform momentum 7 (canonical trainer only), d192
    "int-d192-uni"   = @{ script = (Join-Path $root "jako_train_split.py"); epochs = 30
        args = $intCommon + @("--dim", "192", "--ffn-hidden", "384", "--momentum-bits", "7") }
    # integer d96 (the plateaued baseline, for reference)
    "int-d96-split"  = @{ script = (Join-Path $root "jako_train_split.py"); epochs = 40
        args = $intCommon + @("--dim", "96", "--ffn-hidden", "192", "--momentum-bits", "7", "--source-momentum-bits", "0") }
    # float Adam + cross-entropy ceilings (same split)
    "float-ceiling-d192"      = @{ script = (Join-Path $here "float_ceiling.py"); epochs = 30; args = @("--dim", "192") }
    "float-ceiling-d384"      = @{ script = (Join-Path $here "float_ceiling.py"); epochs = 30; args = @("--dim", "384") }
    # full-length corpus (max_tok 32 / max_len 40 -> 209,979 / 11,051 pairs)
    "float-ceiling-d384-long" = @{ script = (Join-Path $here "float_ceiling.py"); epochs = 20; args = @("--dim", "384", "--max-tok", "32", "--max-len", "40") }
    "float-ceiling-d512-long" = @{ script = (Join-Path $here "float_ceiling.py"); epochs = 20; args = @("--dim", "512", "--max-tok", "32", "--max-len", "40") }
    # integer BlockSGD with the integer base-2 cross-entropy head (exact per-row normalisation), d192
    "int-d192-ce2"   = @{ script = (Join-Path $root "jako_train_split.py"); epochs = 30
        args = $intCommon + @("--dim", "192", "--ffn-hidden", "384", "--momentum-bits", "7", "--source-momentum-bits", "0",
                              "--head", "ce2", "--ce2-shift", "0", "--ce2-cap", "20", "--ce2-norm", "exact") }
    # the same with wider weights: int8 pins at the +-127 clamp under BlockSGD's ramp (max_weight 128 from
    # epoch 0); 10-bit reaches the +-511 clamp by epoch 4 and still climbs; 12-bit has headroom (max 734)
    # (argparse keeps the last --weight-bits, so the override after $intCommon wins)
    "int-d192-w10-ce2" = @{ script = (Join-Path $root "jako_train_split.py"); epochs = 30
        args = $intCommon + @("--dim", "192", "--ffn-hidden", "384", "--weight-bits", "10", "--momentum-bits", "7", "--source-momentum-bits", "0",
                              "--head", "ce2", "--ce2-shift", "0", "--ce2-cap", "20", "--ce2-norm", "exact") }
    "int-d192-w12-ce2" = @{ script = (Join-Path $root "jako_train_split.py"); epochs = 30
        args = $intCommon + @("--dim", "192", "--ffn-hidden", "384", "--weight-bits", "12", "--momentum-bits", "7", "--source-momentum-bits", "0",
                              "--head", "ce2", "--ce2-shift", "0", "--ce2-cap", "20", "--ce2-norm", "exact") }
    # float mirror of that head (base-2 CE, exact Z, cap 20, exp2 attention, Adam)
    "float-ce2-exp2-adam"      = @{ script = (Join-Path $here "float_ablation.py"); epochs = 12; args = @("--dim", "192", "--attn", "exp2", "--opt", "adam", "--ce2", "--ce2-shift", "0", "--ce2-cap", "20") }
    # float ablation ladder: the integer path's constraints one at a time
    "float-hinge-softmax-adam" = @{ script = (Join-Path $here "float_ablation.py"); epochs = 20; args = @("--dim", "192", "--loss", "hinge", "--attn", "softmax", "--opt", "adam") }
    "float-hinge-exp2-adam"    = @{ script = (Join-Path $here "float_ablation.py"); epochs = 20; args = @("--dim", "192", "--loss", "hinge", "--attn", "exp2", "--opt", "adam") }
    "float-ce-exp2-adam"       = @{ script = (Join-Path $here "float_ablation.py"); epochs = 20; args = @("--dim", "192", "--loss", "ce", "--attn", "exp2", "--opt", "adam") }
}

if ($List -or -not $jobs.Contains($Job)) {
    "jobs:"; $jobs.Keys | ForEach-Object { "  $_" }
    if ($Job -and -not $jobs.Contains($Job)) { throw "unknown job $Job" }
    return
}
$j = $jobs[$Job]
$dir = Join-Path $runs $Job
New-Item -ItemType Directory -Force $dir | Out-Null
$ep = if ($Epochs -gt 0) { $Epochs } else { $j.epochs }
$isInt = $j.script -like "*jako_train_split.py"
if ($isInt) {
    $args = @($j.script) + $j.args + @("--epochs", "$ep", "--threads", "$Threads",
        "--save-ckpt", (Join-Path $dir "$Job.pt"), "--out", (Join-Path $dir "$Job.json"),
        "--probe-file", (Join-Path $dir "$Job`_probe.txt"), "--run-tag", "$Job ")
} else {
    $args = @("-X", "utf8", "-u", $j.script) + $j.args + @("--epochs", "$ep", "--threads", "$Threads",
        "--device", $Device, "--out", (Join-Path $dir "$Job.json"))
}
$p = Start-Process -FilePath $Python -ArgumentList $args -WorkingDirectory $root `
    -RedirectStandardOutput (Join-Path $dir "$Job.log") -RedirectStandardError (Join-Path $dir "$Job.err") `
    -PassThru -WindowStyle Hidden
"$Job started, PID $($p.Id); log: $(Join-Path $dir "$Job.log")"
