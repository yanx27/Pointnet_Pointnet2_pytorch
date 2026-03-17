# PointNet++ Baseline for LiDAR Dust Filtering

This repository contains a streamlined, highly customized implementation of the PointNet++ semantic segmentation architecture. It is specifically tailored to serve as a robust learning-based baseline for evaluating LiDAR point cloud dust filtering methods on the **LiDARDustX** dataset.

## 🚀 Key Features Tailored for Adverse Weather

Unlike standard PointNet++ implementations built for indoor CAD models (like ModelNet40), this repository has been optimized for the harsh realities of outdoor, heterogeneous LiDAR data:
* **Dynamic Density Handling:** Includes a custom `LiDARDustXDataset` loader that performs random downsampling and padding to dynamically handle extreme point density variations (20k to 120k points per frame) across heterogeneous sensors (e.g., OUSTER, M1, LS128).
* **Multimodal Feature Fusion:** Natively integrates LiDAR intensity (reflectance) as a critical 4th input channel alongside (x, y, z) coordinates, which is essential for identifying low-reflectance dust aerosols.
* **Class Imbalance Mitigation:** Utilizes a weighted Cross-Entropy Loss mechanism during training to counteract the extreme background-to-dust imbalance (often >99% clean points vs. <1% dust points).
* **Minimalist Architecture:** Stripped of all irrelevant classification and part-segmentation bloat, retaining only the core semantic segmentation engines (`pointnet2_sem_seg.py` and `pointnet2_utils.py`).

## 📁 Repository Structure

* `models/`: Contains the core PointNet++ neural network definitions and sampling/grouping utilities (FPS, Ball Query).
* `dust_dataset.py`: The custom PyTorch `Dataset` class for loading, normalizing, and formatting LiDARDustX data.
* `train_baseline.py`: The primary training loop script configured for the LY150 sensor domain.

## 🛠️ Usage

This codebase is designed to run within a Dockerized PyTorch environment (`nvcr.io/nvidia/pytorch:23.10-py3`). 

To initiate training on the base sensor domain (e.g., LY150):
```bash
python train_baseline.py
