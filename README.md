# DCASE2025 - Task 1 - Inference Package

Contact: **Florian Schmid** (florian.schmid@jku.at), *Johannes Kepler University Linz*

Official Task Description:  
🔗 [DCASE Website](https://dcase.community/challenge2025/task-low-complexity-acoustic-scene-classification-with-device-information) 
📄 [Task Description Paper (arXiv)](https://arxiv.org/pdf/2505.01747) 


## Device-Aware Inference for Low-Complexity Acoustic Scene Classification

This repository contains the **inference package** for DCASE 2025 Task 1 and is designed to support:
- **Reproducible and open research** through a standardized Python inference interface  
- **Automatic complexity checking** (MACs, Params) for each device  
- **Simple and correct model evaluation** on the evaluation set, including sanity checks on the test set  

The package is implemented as an installable Python module and provides a clean API for generating predictions and evaluating model complexity using a pre-trained model.

**Participants of Task 1 are required to submit a link to their inference code package on GitHub.  
The inference code package must implement the API outlined in this README file.**


---

## File Overview

The repository includes the following key components:

```
.
├── Schmid_CPJKU_task1/
│ ├── Schmid_CPJKU_task1_1.py # Submission Module: Main inference interface, implements API
│ ├── models/ # Model architecture and device container
│ ├── resources/ # Dummy file and test split CSV
│ ├── ckpts/ # Model checkpoints
├── complexity.py # Helper functions for complexity measurements
├── test_complexity.py # Script to check MACs and Params
├── evaluate_submission.py # Run predictions on test/eval sets
├── requirements.txt # Required Python packages
├── setup.py # Installable Python package
```


Participants are allowed to submit up to **four inference packages**. This baseline repository includes a single submission module named `Schmid_CPJKU_task1_1`.  
Additional submissions would be added as:

- `Schmid_CPJKU_task1_2`
- `Schmid_CPJKU_task1_3`
- `Schmid_CPJKU_task1_4`

---

## 🧩 Inference API: Required Functions

Each submission module **must implement** the following four functions to ensure compatibility with the official DCASE 2025 evaluation scripts.  
Example implementations of these functions are implemented for the baseline system in file `Schmid_CPJKU_task1_1.py`.

---

```
predict(
  file_paths: List[str], 
  device_ids: List[str], 
  model_file_path: Optional[str] = None
) -> Tuple[List[Tensor], List[str]]
```

Run inference on a list of audio files.

**Args:**
- `file_paths`: List of `.wav` file paths to predict.
- `device_ids`: List of device IDs corresponding to each file (must match `file_paths` length).
- `model_file_path`: Optional path to a model checkpoint (`.ckpt`). If `None`, the default packaged checkpoint must be used.

**Returns:**
- A tuple `(logits, class_order)`:
  - `logits`: List of tensors of shape `[n_classes]`, one per file, in the original input order.
  - `class_order`: List of class names (e.g., `["airport", "bus", ..., "tram"]`) corresponding to the class index positions in the output logits.

---

```
load_model(
  model_file_path: Optional[str] = None
) -> torch.nn.Module
```

Load the pretrained model used for inference.

**Args:**
- `model_file_path`: Optional path to a model checkpoint (`.ckpt`). If `None`, the default packaged checkpoint must be used.

**Returns:**
- A PyTorch model object that supports inference and can be passed to `load_inputs()` and `get_model_for_device()`.

---

```
load_inputs(
  file_paths: List[str],
  device_ids: List[str],
  model: torch.nn.Module
) -> List[Tensor]
```

Prepare inputs for inference by converting raw waveform audio into model-ready input tensors.

**Args:**
- `file_paths`: List of `.wav` file paths.
- `device_ids`: List of corresponding device IDs.
- `model`: An instance of the model returned by `load_model()`.

**Returns:**
- List of model input tensors, in the same order as `file_paths`.

---

```
get_model_for_device(
  model: torch.nn.Module, 
  device_id: str
) -> torch.nn.Module
```

Return the submodel corresponding to a specific recording device.

**Args:**
- `model`: The model instance returned by `load_model()`.
- `device_id`: The string identifier of the recording device (e.g., `"a"`, `"b"`, `"s1"`, ..., or `"unknown"`).

**Returns:**
- A PyTorch `nn.Module` that contains only the submodel for the specified device.

---


## Getting Started

Follow the steps below to prepare and test your DCASE 2025 Task 1 inference submission:

1. Clone this repository:

```bash
git clone https://github.com/CPJKU/dcase2025_task1_inference
cd dcase2025_task1_inference
```

2. Rename the package (`Schmid_CPJKU_task1`) using your official submission label (see [here](https://dcase.community/challenge2024/submission#submission-label) for informations on how to form your submission label).
3. Rename the module (`Schmid_CPJKU_task1_1` inside package) using your submission label + the submission index (`1` in the example). You may submit up to four modules with increasing submission index (`1` to `4`).
4. Create a Conda environment: `conda create -n d25_t1_inference python=3.13`. Activate your conda environment.
5. Install your package locally `pip install -e .`. Don't forget to adapt the `requirements.txt` file later on if you add additional dependencies.
6. Implement your submission module(s) by defining the required API functions (see above). 
7. Verify that your models comply with the complexity limits (MACs, Params):

```python test_complexity.py --submission_name <submission_label> --submission_index <submission_number>```

8. Download the evaluation set (to be released on June 1st). 
9. Evaluate your submissions on the test split and generate evaluation set predictions:
```
python evaluate_submission.py \
    --submission_name <submission_label> \
    --submission_index <submission_number> \
    --dev_set_dir /path/to/TAU-2022-development-set/ \
    --eval_set_dir /path/to/TAU-2025-eval-set/
```

After successfully running the scripts in steps 8. and 9., a folder `predictions` will be generated inside `dcase2025_task1_inference`:

```
predictions/
└── <submission_label>_<submission_index>/
    ├── output.csv             # Evaluation set predictions (submit this file)
    ├── model_state_dict.pt    # Model weights (optional, for reproducibility)
    ├── test_accuracy.json     # Test set accuracy (sanity check only)
    └── complexity.json        # MACs and parameter memoyr per device model
└── <submission_label>_<submission_index>/ # up to four submissions
.
.
.
```
