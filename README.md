# AI-Powered 3D Segmentation of Lower Spinal Vertebrae

Final Year Project (FYP) for automated 3D segmentation of lower spinal vertebrae from MRI scans using a custom **3D U-Net** built with TensorFlow/Keras.

## Overview

This repository provides notebooks to train and run inference with a 3D U-Net model on NIfTI (`.nii`) MRI volumes. Volumes are resized to `(32, 256, 256)` and normalized before being passed to the network. The model outputs a binary segmentation mask for lower spinal vertebrae.

### Model Architecture

| Component | Details |
|-----------|---------|
| Architecture | 3D U-Net (3 encoder / 3 decoder levels) |
| Input shape | `(32, 256, 256, 1)` |
| Output | Binary mask `(32, 256, 256, 1)` with sigmoid activation |
| Encoder filters | 64 → 128 → 256 |
| Bridge | 1024 filters |
| Decoder filters | 256 → 128 → 64 |
| Loss | Binary cross-entropy + Dice loss |
| Optimizer | Adam |
| Metric | Dice coefficient |

## Repository Structure

```
├── Dataset/
│   ├── Images/          # 23 MRI volumes (Img_01.nii – Img_23.nii)
│   └── Labels/          # Corresponding ground-truth masks (Img_01_Labels.nii – …)
├── Notebooks/
│   ├── Model Training.ipynb   # Train the 3D U-Net on preprocessed data
│   └── inference.ipynb        # Load weights and segment a single MRI volume
└── README.md
```

## Model Weights

Pre-trained weights are available for download:

**[Download model weights (Google Drive)](https://drive.google.com/file/d/1c55PafjbbNes6aDKTilp5ZPQ5JDBqNrS/view?usp=sharing)**

Expected filename: `model_checkpoint_256_100_epochs_full_unet256_1024_3_layers.keras`

## Requirements

Install the following Python packages:

```
tensorflow
keras
nibabel
scikit-image
numpy
matplotlib
pandas
scikit-learn
SimpleITK
```

For the training notebook on Kaggle, `gdown` is also used to fetch the preprocessed dataset.

## Notebooks

### 1. Model Training (`Notebooks/Model Training.ipynb`)

Trains the 3D U-Net on a preprocessed dataset stored as a pickle file.

**Workflow:**

1. Download the preprocessed dataset (`All_images_tuple_32256256.pkl`) via Google Drive.
2. Load train/validation splits from the pickle file.
3. Build and compile the 3D U-Net.
4. Train for **100 epochs** with `batch_size=1`, using:
   - `ModelCheckpoint` — saves best weights by validation loss
   - `ReduceLROnPlateau` — reduces learning rate on plateau (patience 20)
   - `EarlyStopping` — stops early if validation loss stalls (patience 15)
5. Evaluate on random validation samples with slice-wise visualizations (X, Y, Z axes) and Dice scores.

**Training configuration:**

| Parameter | Value |
|-----------|-------|
| Epochs | 100 |
| Batch size | 1 |
| Input shape | `(32, 256, 256, 1)` |
| Checkpoint name | `model_checkpoint_256_V2_100_epochs_full_unet256_1024_3_layers.keras` |

> **Note:** This notebook is configured for **Kaggle** (`/kaggle/working/` paths). Adjust paths if running locally.

### 2. Inference (`Notebooks/inference.ipynb`)

Runs segmentation on a single NIfTI MRI volume using pre-trained weights.

**Workflow:**

1. Set the path to your input `.nii` file.
2. Load and preprocess the volume (normalize to `[0, 1]`, resize to `(32, 256, 256)`).
3. Rebuild the 3D U-Net architecture (must match training).
4. Load pre-trained weights from the `.keras` checkpoint.
5. Run `model.predict()` to obtain the segmentation mask.

**Preprocessing function:**

```python
def load_nifti_image(image_path, target_shape=(32, 256, 256)):
    image = nib.load(image_path).get_fdata(dtype=np.float32)
    image = image / np.max(image)
    image_resized = resize(image, target_shape, mode='constant', preserve_range=True)
    image_resized = image_resized[..., np.newaxis]
    return image_resized
```

**Before running inference locally:**

1. Download model weights from the Google Drive link above.
2. Update `image_path` to point to your MRI file (e.g. `Dataset/Images/Img_01.nii`).
3. Update `file_path` to the location of the downloaded `.keras` weights file.

## Dataset

The `Dataset/` folder contains **23 paired MRI volumes and labels** in NIfTI format:

- **Images:** `Dataset/Images/Img_XX.nii`
- **Labels:** `Dataset/Labels/Img_XX_Labels.nii`

Volumes are resized to `(32, 256, 256)` during preprocessing. The identifier `32256256` in filenames and pickle data refers to this target resolution.

## Quick Start (Inference)

1. Clone this repository.
2. Install dependencies (`pip install tensorflow nibabel scikit-image numpy matplotlib`).
3. Download model weights from Google Drive.
4. Open `Notebooks/inference.ipynb`.
5. Set `image_path` and `file_path` to your local paths.
6. Run all cells — the output mask shape will be `(1, 32, 256, 256, 1)`.

## License

This project was developed as a Final Year Project. Please contact the authors for usage terms.
