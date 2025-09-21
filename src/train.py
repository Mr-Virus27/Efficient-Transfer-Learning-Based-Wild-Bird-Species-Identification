import numpy as np
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from src.model import build_model
import json
import os

def main(features='mfcc_features.npy', labels='labels.npy', label_names='label_names.json', out_model='bird_vgg16_model.h5'):
    if not os.path.exists(features) or not os.path.exists(labels):
        raise FileNotFoundError('Feature or label files not found. Run src/dataset.py to prepare them.')

    X = np.load(features)
    y = np.load(labels)
    with open(label_names, 'r') as f:
        labels_list = json.load(f)

    # ensure shape: (N, H, W, 3)
    if X.ndim != 4:
        raise ValueError('Expected X to have 4 dims (N,H,W,3).')

    y_cat = to_categorical(y, num_classes=len(labels_list))
    X_train, X_test, y_train, y_test = train_test_split(X, y_cat, test_size=0.2, random_state=42, stratify=y)

    model = build_model(input_shape=X_train.shape[1:], num_classes=y_train.shape[1], lr=1e-4)
    history = model.fit(X_train, y_train,
                        validation_data=(X_test, y_test),
                        epochs=10,
                        batch_size=16)
    model.save(out_model)
    # save test split for evaluation
    np.save('X_test.npy', X_test)
    np.save('y_test.npy', y_test)
    print(f"Model saved to {out_model}")

if __name__ == '__main__':
    main()
