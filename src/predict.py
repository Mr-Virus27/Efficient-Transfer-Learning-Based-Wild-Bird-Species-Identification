import numpy as np
from tensorflow.keras.models import load_model
import json
from src.preprocess import extract_mfcc

def predict_file(model_path, audio_path, label_names='label_names.json'):
    model = load_model(model_path)
    mfcc = extract_mfcc(audio_path)
    X = mfcc[None, ...]  # (1, H, W)
    X = X[..., None]
    X = np.repeat(X, 3, axis=-1)
    preds = model.predict(X)
    idx = preds.argmax(axis=1)[0]
    with open(label_names, 'r') as f:
        labels = json.load(f)
    return labels[idx], float(preds.max())

if __name__ == '__main__':
    import sys
    if len(sys.argv) < 3:
        print('Usage: python src/predict.py model.h5 audio.wav')
    else:
        label, conf = predict_file(sys.argv[1], sys.argv[2])
        print(f'Prediction: {label} (confidence={conf:.3f})')
