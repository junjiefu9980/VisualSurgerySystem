# Visual-based Surgery Skill Evaluation System

## Dataset Source Statement

This project uses the **SurgPose** dataset, a public dataset for articulated robotic surgical tool pose estimation and tracking.
SurgPose provides stereo surgical videos and semantic keypoint annotations for instrument pose estimation and trajectory analysis.

Dataset repository: https://github.com/zijianwu1231/SurgPose  
Paper: https://arxiv.org/abs/2502.11534

## Overview

This repository implements **a vision-based surgical skill evaluation system** for **video-based surgical skill assessment**.
The main design principle is **stabilization as a prerequisite for evaluation**: the system first builds reliable 2D and 3D trajectory outputs, then uses policy routing and Temporal Smoothing before generating case-level feedback.

The current packaged run uses stereo endoscopic video pairs / left-right views when triangulation is feasible. The default pipeline reuses cached YOLO and MonSter outputs when the heavy rebuild inputs are not available, rebuilds `Tri` locally, refreshes evaluation tables, and then generates simplified agent outputs.

## 1. Environment Configuration

Create and activate a Python environment, then install the default dependencies:

```bash
pip install -r requirements.txt
```

The default reproducible run does NOT require YOLO training or MonSter targeted rebuild. It can reuse:

- `output/detections/2d_results.csv`
- `output/detections/3d_monster_results.csv`

For full data or rebuild inputs, refer to the SurgPose GitHub link above.

## 2. Main Reproduction Commands

Run the visual pipeline:

```bash
python src/pipeline/run_pipeline.py
```

Run the simplified agent outputs:

```bash
python src/agent/run_agent_pipeline.py
```

## 3. Output & Summary

The visual pipeline follows: 2D observation -> hybrid 3D reconstruction -> policy-based route selection (`tri`, `monster`, `2d`, `None`) -> temporal smoothing -> final case-level summary.

The agent is used after Policy-Based Selection and Temporal Smoothing as a case-level evidence organisation module.
It produces three layers of evidence: case-level summary, structured event evidence and natural-language feedback. It automatically groups cases by route and selects representative cases for qualitative review. It then draws only the final chosen method for each selected case.

The selected cases are recorded in `output/agent/summary.csv`.
