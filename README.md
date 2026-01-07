# Bring Your Own Container (BYOC) – Amazon SageMaker Demo

## Overview

This project demonstrates how to **train models on Amazon SageMaker using a custom Docker container** (Bring Your Own Container, BYOC).

Instead of relying on SageMaker’s built-in frameworks, this approach gives full control over:
- The runtime environment
- Dependency versions
- Training entrypoints
- Container behavior

This mirrors how production ML teams deploy custom training stacks when default SageMaker images are insufficient.

---

## What This Project Demonstrates

- Building a **custom training container** compatible with SageMaker
- Using SageMaker’s training APIs with **user-defined Docker images**
- Separating **container logic** from **training orchestration**
- Running containerized training jobs on managed infrastructure

This project focuses on **infrastructure and integration**, not model performance.

---

## Repository Structure

```
.
├── container/
│   ├── Dockerfile          # Custom training image
│   └── train.py            # Container entrypoint
│
├── training/
│   └── train.py            # SageMaker training job launcher
│
├── README.md
└── .gitignore
```

### `container/`
Defines the Docker image used by SageMaker:
- Installs dependencies
- Defines the training entrypoint
- Handles input/output paths expected by SageMaker

### `training/`
Launches the SageMaker training job:
- References the custom image
- Configures instance type, role, and job parameters
- Submits the job to SageMaker

---

## Execution Flow

1. **Build the Docker image**
2. **Push the image to Amazon ECR**
3. **Launch a SageMaker training job** referencing that image
4. SageMaker:
   - Pulls the image
   - Mounts training data
   - Executes the container entrypoint
   - Collects outputs

This flow mirrors production BYOC workflows.

---

## Why BYOC?

SageMaker BYOC is useful when:
- You need nonstandard system libraries
- You require custom CUDA / framework versions
- You want identical local and cloud training environments
- Built-in SageMaker containers are too restrictive

This pattern is common in real-world ML infrastructure.

---

## How to Run (High-Level)

1. Build and push the container to ECR  
2. Update the image URI in `training/train.py`
3. Run the training launcher:
   ```bash
   python training/train.py
   ```

AWS credentials and permissions must already be configured.

---

## Limitations

- No hyperparameter tuning
- No distributed or multi-GPU training
- Minimal logging and monitoring
- Assumes a trusted execution environment

These are intentional to keep the demo focused on BYOC mechanics.

---

## Scope Clarification

This project is **not**:
- A full production training pipeline
- A model benchmarking project
- A reusable SageMaker framework

It **is**:
- A BYOC reference implementation
- A SageMaker integration example
- An ML infrastructure learning artifact
