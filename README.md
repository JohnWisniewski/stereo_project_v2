# Stereo Matching Project
# Author: John Wisniewski
# Date: 4/26/2025

---

## Project Description

This project implements a stereo analysis system for computing disparity maps 
from stereo image pairs, using both region-based and feature-based methods.

Features include:
- Region-based matching with SAD, SSD, NCC cost metrics
- Feature-based matching using Harris Corner Detector (with R value as descriptor)
- Multi-resolution coarse-to-fine disparity estimation
- Left-to-right consistency checking
- Gap filling for invalid disparities
- Modular Python code following clean structure

---

## Instructions to Run

Environment:
- Python 3.10+
- Install dependencies:

    pip install -r requirements.txt

Execution:
- Set permissions:

    Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass

- Then run all experiments:

    .\run_all.ps1

This script automatically runs 5 experiments and saves the results.

---

## Experiments Conducted (5 Total)

### 1. Cones — Region-Based Matching
- Left Image: `bench_pairs/cones_left.png`
- Right Image: `bench_pairs/cones_right.png`
- Method: Region
- Metric: NCC
- Template Size: 11×11
- Max Disparity: 70
- Multi-resolution: Enabled
- Output Directory: `results_cones_region/`

### 2. Cones — Feature-Based Matching
- Left Image: `bench_pairs/cones_left.png`
- Right Image: `bench_pairs/cones_right.png`
- Method: Feature
- Metric: SSD
- Patch Size: 9×9
- Max Disparity: 70
- Output Directory: `results_cones_feature/`

### 3. Teddy — Region-Based Matching
- Left Image: `bench_pairs/teddy_left.png`
- Right Image: `bench_pairs/teddy_right.png`
- Method: Region
- Metric: SAD
- Template Size: 9×9
- Max Disparity: 70
- Multi-resolution: Enabled
- Output Directory: `results_teddy_region/`

### 4. Tsukuba — Region-Based Matching
- Left Image: `bench_pairs/tsukuba_left.png`
- Right Image: `bench_pairs/tsukuba_right.png`
- Method: Region
- Metric: NCC
- Template Size: 7×7
- Max Disparity: 16
- Multi-resolution: Enabled
- Output Directory: `results_tsukuba_region/`

### 5. Tsukuba — Feature-Based Matching
- Left Image: `bench_pairs/tsukuba_left.png`
- Right Image: `bench_pairs/tsukuba_right.png`
- Method: Feature
- Metric: NCC
- Patch Size: 11×11
- Max Disparity: 16
- Output Directory: `results_tsukuba_feature/`

---

## Outputs

Each experiment produces 3 output images:
- *_raw.png — Disparity map without validation
- *_valid.png — After left-right consistency checking
- *_filled.png — Final disparity map with gaps filled

All output images are located under their respective `results_*` folders.

---

## Notes

- Feature-based matching uses Harris corner response (R value) for matching.
- Region-based matching uses dense pixelwise template matching.
- Gap filling uses adaptive neighborhood averaging.
- No runtime exceptions; modular design with clear documentation and comments.

---
