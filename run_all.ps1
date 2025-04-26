# --- run_all.ps1 -----------------------------------------------------------
$ErrorActionPreference = "Stop"
$env:PYTHONPATH = (Get-Location)

function Run {
    param([string]$args)
    Write-Host ">>> python $args"
    python $args
    Write-Host "-----`n"
}

Run "-m stereo.run_demo --left .\bench_pairs\cones_left.png --right .\bench_pairs\cones_right.png --method region --metric NCC --block 11 --max_disp 70 --multires --outdir results_cones_region"

Run "-m stereo.run_demo --left .\bench_pairs\cones_left.png --right .\bench_pairs\cones_right.png --method feature --metric SSD --block 9 --max_disp 70 --outdir results_cones_feature"

Run "-m stereo.run_demo --left .\bench_pairs\teddy_left.png --right .\bench_pairs\teddy_right.png --method region --metric SAD --block 9 --max_disp 70 --multires --outdir results_teddy_region"

Run "-m stereo.run_demo --left .\bench_pairs\tsukuba_left.png --right .\bench_pairs\tsukuba_right.png --method region --metric NCC --block 7 --max_disp 16 --multires --outdir results_tsukuba_region"

Run "-m stereo.run_demo --left .\bench_pairs\tsukuba_left.png --right .\bench_pairs\tsukuba_right.png --method feature --metric NCC --block 11 --max_disp 16 --outdir results_tsukuba_feature"
# --------------------------------------------------------------------------