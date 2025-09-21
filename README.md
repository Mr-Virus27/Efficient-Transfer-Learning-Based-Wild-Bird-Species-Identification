<<<<<<< HEAD
# Efficient-Transfer-Learning-Based-Wild-Bird-Species-Identification
This project uses transfer learning with pre-trained CNNs to efficiently identify wild bird species from images. By leveraging existing models, it achieves high accuracy with limited data, reducing training time and computational resources. The system aids wildlife monitoring, research, and conservation efforts.
=======
# Efficient Transfer Learning-based Wild Bird Species Identification

This repository is a working implementation derived from the Bachelor's thesis:
**Efficient Transfer Learning-based Wild Bird Species Identification** (May 2024).

## Contents
- `src/` — Python source code:
  - `preprocess.py` — audio loading & MFCC extraction
  - `dataset.py` — dataset helper and splitting utilities
  - `model.py` — VGG16 transfer-learning model builder
  - `train.py` — training script (expects precomputed features or raw files)
  - `evaluate.py` — evaluation script (prints classification report & confusion matrix)
  - `predict.py` — single-audio prediction helper
  - `app.py` — minimal Flask app to upload audio and get prediction
- `notebooks/` — suggested notebooks (placeholder)
- `requirements.txt` — Python dependencies
- `data/` — not included (link to Kaggle dataset recommended)

## Quickstart (local)
1. Create a Python environment (recommended: conda or venv).
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Place your audio dataset (or small sample) inside `data/` and follow the usage comments
   in `src/dataset.py` to generate `mfcc_features.npy` and `labels.npy`.
4. Train:
   ```
   python src/train.py
   ```
5. Evaluate:
   ```
   python src/evaluate.py
   ```
6. Run the Flask demo:
   ```
   python src/app.py
   ```
   Then open http://127.0.0.1:5000 to upload an audio file and see predictions.

## Notes
- The actual Kaggle BirdCLEF dataset is large and not included. Instead, add a small sample
  for quick testing or point scripts to your dataset path.
- This implementation uses spectrogram/MFCC-based features converted to an image-like
  shape to reuse ImageNet-pretrained VGG16. You can adapt model.py to other architectures.

## License
MIT
>>>>>>> ac8a96d (Initial commit with project files)
