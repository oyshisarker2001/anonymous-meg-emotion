
# TEAM-Net: Cross-Subject Emotion Recognition from MEG

This repository contains code for cross-subject emotion recognition using magnetoencephalography (MEG) signals from the OpenNeuro ds007640 dataset. The proposed framework, TEAM-Net (Task-Adaptive Emotion Affective MEG Network), models arousal and valence independently using task-specific architectures and evaluates performance under a leave-one-subject-out (LOSO) protocol.

---

## Installation

Install required dependencies:

```bash
pip install -r requirements.txt
```

---

## Dataset

This project uses the OpenNeuro ds007640 dataset:

https://openneuro.org/datasets/ds007640/versions/1.0.0

### Setup

1. Download the dataset from the link above.
2. Place it in the project directory as:

```
ds007640-download/
```

---

## Preprocessing

Before running the models, preprocess the data:

```bash
python preprocessingData.py
```

This step extracts event-aligned MEG segments and generates the processed dataset required for training.

---

## File Structure

### Preprocessing

* `preprocessingData.py`
  Main preprocessing script that generates processed trial-level MEG data.

* `event_checking.py`
  Inspects MEG trigger channels, annotations, and event distributions.

* `event_find.py`
  Displays candidate event IDs and their chronological order.

* `preprocessing.py`
  Exploratory script for validating label-event alignment.

---

### Main Models

* `train_arousal_loso.ipynb`
  Arousal classification model using CNN–BiLSTM with attention and gated fusion under LOSO evaluation.

* `train_valence_loso.ipynb`
  Valence classification model using multi-scale CNN with squeeze-and-excitation and gated pooling under LOSO evaluation.

---

### Data and Outputs

* `preprocessed_dataset/`
  Processed MEG trial data generated after preprocessing.

* `summary_all_subjects_sessions.csv`
  Summary of subjects and sessions.

* Output files (generated during training):

  * model checkpoints (`.pt`)
  * prediction arrays (`.npy`)
  * training history (`.json`)
  * performance curves (`.png`)

---

## How to Run

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Preprocess data

```bash
python preprocessingData.py
```

### 3. Run arousal model

Open and run:

```
train_arousal_loso.ipynb
```

### 4. Run valence model

Open and run:

```
train_valence_loso.ipynb
```

---

## Method Overview

* Event-aligned segmentation of continuous MEG signals
* Window-based representation of trial data
* Task-specific modeling:

  * Arousal → CNN–BiLSTM with attention and gated fusion
  * Valence → Multi-scale CNN with squeeze-and-excitation and gated pooling
* Imbalance-aware hybrid loss (BCE + Soft-F1)
* Leave-one-subject-out (LOSO) evaluation

---

## Reproducibility

This repository provides:

* Complete preprocessing pipeline
* Model training and evaluation notebooks
* Saved outputs for analysis

All components are included to support reproducibility of the reported results.

---

## Notes

* The dataset is not included in this repository due to size constraints.
* Ensure the dataset path matches the expected directory structure.
* Each notebook performs training, evaluation, and saves results automatically.
## END