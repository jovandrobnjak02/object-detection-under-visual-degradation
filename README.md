# Object Detection Under Visual Degradation

A Master's thesis comparing four object detection architectures for robustness under adverse driving conditions (rain, nighttime) using the BDD100K dataset.

## Project Description

This project benchmarks four architectures spanning the evolution from CNN-based to full-transformer models:

| Model | Architecture | Framework |
|-------|-------------|-----------|
| YOLOv11 | CNN-first, minimal attention | Ultralytics |
| YOLOv12 | CNN backbone + attention | Ultralytics |
| RT-DETR | Full transformer (no pretraining) | Ultralytics |
| RF-DETR | Full transformer + DINOv2 backbone | Roboflow (`rfdetr`) |

All models are trained exclusively on clear-weather images from BDD100K, then evaluated on three test splits: **clear**, **rain**, and **nighttime**. The goal is to measure how architectural choices affect robustness degradation under distribution shift.

## Setup (Google Colab Pro + VS Code Extension)

1. Install the [Colab extension for VS Code](https://marketplace.visualstudio.com/items?itemName=Google.colab-debugger-for-vscode)
2. Connect to a Colab Pro runtime (NVIDIA A100 recommended)
3. Mount your Google Drive or upload the BDD100K dataset to the runtime
4. Copy `.env.example` to `.env` and fill in your dataset paths
5. Run notebook `01_setup_and_data.ipynb` to verify the environment

## Running the Experiments

Execute notebooks in order:

```
01_setup_and_data.ipynb     → verify environment, load and inspect BDD100K
02_train_yolov11.ipynb      → train YOLOv11 on clear-weather split
03_train_yolov12.ipynb      → train YOLOv12 on clear-weather split
04_train_rtdetr.ipynb       → train RT-DETR on clear-weather split
05_train_rfdetr.ipynb       → train RF-DETR on clear-weather split
06_evaluate_clear.ipynb     → evaluate all 4 models on clear test set
07_evaluate_adverse.ipynb   → evaluate all 4 models on rain + nighttime
08_analysis_and_plots.ipynb → generate comparison tables and thesis figures
```

## Dataset

**BDD100K** — Berkeley DeepDrive 100K  
100,000 driving videos with object detection annotations across diverse weather and lighting conditions.

- Download: https://bdd-data.berkeley.edu/
- 10 detection classes: pedestrian, rider, car, truck, bus, train, motorcycle, bicycle, traffic light, traffic sign
- Splits used: clear train/val for training; clear/rain/nighttime test for evaluation

## Why These Four Models?

The four models form a deliberate architectural ladder:
- **YOLOv11** establishes the CNN baseline with speed-optimised detection heads
- **YOLOv12** adds self-attention to the CNN backbone to test whether attention improves robustness
- **RT-DETR** replaces the CNN backbone with a full transformer, removing inductive CNN biases
- **RF-DETR** adds DINOv2 pretraining on top of the transformer architecture, testing whether large-scale pretraining transfers robustness
