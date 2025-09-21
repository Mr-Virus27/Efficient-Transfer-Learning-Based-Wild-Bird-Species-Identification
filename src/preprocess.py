import librosa
import numpy as np
import os

def extract_mfcc(file_path, n_mfcc=40, max_pad_len=174, sr=22050):
    '''
    Load an audio file and extract MFCC features with padding/truncation.
    Returns a (n_mfcc, max_pad_len) shaped numpy array.
    '''
    audio, sample_rate = librosa.load(file_path, sr=sr, res_type='kaiser_fast')
    mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=n_mfcc)
    pad_width = max_pad_len - mfccs.shape[1]
    if pad_width > 0:
        mfccs = np.pad(mfccs, pad_width=((0, 0), (0, pad_width)), mode='constant')
    else:
        mfccs = mfccs[:, :max_pad_len]
    return mfccs

def batch_extract_from_folder(folder, extensions=['.wav', '.mp3']):
    '''
    Walks through `folder`, extracts MFCC for each audio file and returns arrays.
    Expects subfolders named by class label (one folder per species).
    '''
    X = []
    y = []
    labels = []
    for idx, label in enumerate(sorted(os.listdir(folder))):
        label_path = os.path.join(folder, label)
        if not os.path.isdir(label_path):
            continue
        labels.append(label)
        for fname in sorted(os.listdir(label_path)):
            if not any(fname.lower().endswith(ext) for ext in extensions):
                continue
            path = os.path.join(label_path, fname)
            try:
                mfcc = extract_mfcc(path)
                X.append(mfcc)
                y.append(idx)
            except Exception as e:
                print(f"Failed to process {path}: {e}")
    X = np.array(X)
    y = np.array(y)
    return X, y, labels

if __name__ == '__main__':
    print('Run this file as a module from dataset.py or import the functions.')
