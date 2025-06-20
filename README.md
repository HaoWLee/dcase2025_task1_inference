# dcase2025_task1_inference

My submission for **DCASE 2025 Task 1: Device-Aware Inference for Low-Complexity Acoustic Scene Classification**

**Contact:** Haowen Li ([haowen.li@ntu.edu.sg](mailto:haowen.li@ntu.edu.sg))  
**Affiliation:** Nanyang Technological University

---

##  Official Task Description

🔗 [DCASE Website](https://dcase.community/challenge2025/)  
(Task 1: Low-complexity acoustic scene classification)

---

##  Installation (Requirements)

Create a Python environment (e.g. Python 3.10) and install the following dependencies:

```bash
pip install librosa
            numpy
            torch
            tqdm
            scikit-learn
            pandas
            torchinfo
            av
            h5py
            sox
            soundfile
            torchvision
```
##  File Overview

The repository includes the following key components:
```
.
├── Li_NTU_task1/
│ ├── Li_NTU_task1_1.py # Submission Module: Main inference interface, implements API
│ ├── Li_NTU_task1_2.py # Submission Module: Main inference interface, implements API
│ ├── models/ # Model architecture and device container
│ ├── resources/ # Dummy file and test split CSV
│ ├── ckpts/ # Model checkpoints
├── predictions/
├── complexity.py # Helper functions for complexity measurements
├── test_complexity.py # Script to check MACs and Params
├── evaluate_submission.py # Run predictions on test/eval sets
├── requirements.txt # Required Python packages
└── setup.py # Installable Python package
```
We submit  **two inference packages**. This repository includes two submissions module named `Li_NTU_task1_1` and `Li_NTU_task1_2`.  




