# SOLAR
**Scalable Distributed Spatial Joins through Learning-based Optimization**

This repository contains the source code for the paper **"SOLAR: Scalable Distributed Spatial Joins through Learning-based Optimization."**

---

## Table of Contents
1. [Overview](#overview)
2. [Datasets](#datasets)
3. [Environment](#environment)
4. [Code Structure](#code-structure)
5. [Quick Start](#quick-start)

---

## Overview
**SOLAR** is a system for performing distributed spatial joins efficiently in large-scale cluster environments. It combines:
- An **offline training phase** featuring a Siamese Neural Network model and a Random Forest decision-making model to learn optimal partitioning strategies.
- An **online execution phase** that applies learned models to speed up spatial join tasks by reusing partitioners and reducing overhead.

---

## Datasets
The datasets used in our experiments (e.g., OpenStreetMap data for different regions) are **not** stored directly in this repository due to their large size. You can download them from the Google Drive link below:

- **[Google Drive Link](https://drive.google.com/file/d/157GDBJz-eScnUebNtpDsC-rV-ElMP1zz/view?usp=drive_link)**

After downloading, place the datasets in your preferred local directory (e.g., `datasets/`) or directly upload them to HDFS. For example:

```bash
hdfs dfs -put <local_dataset_path> <hdfs_target_directory>
```
---
## Environment
Experiments were conducted on a Microsoft Azure cluster:
- **2 Head Nodes**: E8 V3 (8 cores, 64 GB RAM)
- **8 Worker Nodes**: E2 V3 (2 cores, 16 GB RAM)
To run SOLAR:
1. Set up a distributed cluster environment (e.g., Microsoft Azure, Google Cloud, or AWS).
2. Ensure all data is stored in HDFS.
---
## Code Structure
- **`/sedona`**
  Contains Apache Sedona’s original source code plus our modifications. This section is primarily written in Java.
  - Main changes are under:
    `sedona/spark.common/src/main/java/org/apache/sedona/core`
- **`/new_pycode`**
  Contains Python code for the SOLAR system:
  - **Offline Training**
    - Siamese Neural Network model
    - Random Forest decision-making model
  - **Online Execution**
    - Code to apply learned models during runtime
- **Entry Point**
  The main entry point for the system is:
  `SedonaProject/src/com/myproject/SedonaApp.java`
---
## Quick Start
1. **Compile Apache Sedona**
    ```bash
    sh compileSedona.sh
    ```
2. **Compile the Entry Point**
    ```bash
    sh compileJava.sh
    ```
3. **Run Experiments**
   - **Varying Distances**
     ```bash
     python3 learning_py/main_var_distance.py
     ```
   - **Varying Training Data Percentage**
     ```bash
     python3 learning_py/main_var_percentage.py
     ```
---
**Contact:**
For any questions or issues, please reach out to yliu786@ucr.edu
---
