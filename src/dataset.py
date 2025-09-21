import numpy as np
from sklearn.model_selection import train_test_split
from src.preprocess import batch_extract_from_folder
import os

def prepare_and_save(data_dir='data', out_feature_file='mfcc_features.npy', out_label_file='labels.npy', out_labels_txt='label_names.json'):
    '''
    Extracts MFCCs from data_dir (expects subfolders per class), saves numpy arrays.
    '''
    X, y, labels = batch_extract_from_folder(data_dir)
    # If using a CNN pretrained on ImageNet, we need a 3-channel image-like input.
    # Here we simply expand to (H, W, 3) by repeating the MFCCs across channels.
    X_img = X[..., None]  # (N, n_mfcc, time, 1)
    X_img = np.repeat(X_img, 3, axis=-1)  # (N, n_mfcc, time, 3)
    np.save(out_feature_file, X_img)
    np.save(out_label_file, y)
    import json
    with open(out_labels_txt, 'w') as f:
        json.dump(labels, f)
    print(f"Saved features to {out_feature_file}, labels to {out_label_file}, and names to {out_labels_txt}")

if __name__ == '__main__':
    # Example: create/prepare features
    if not os.path.exists('data'):
        print('Please add your dataset under the `data/` folder with one folder per class.')
    else:
        prepare_and_save()
