import numpy as np
from tensorflow.keras.models import load_model
from sklearn.metrics import classification_report, confusion_matrix
import json

def main(model_path='bird_vgg16_model.h5', X_test_file='X_test.npy', y_test_file='y_test.npy', label_names='label_names.json'):
    model = load_model(model_path)
    X_test = np.load(X_test_file)
    y_test = np.load(y_test_file)
    with open(label_names, 'r') as f:
        labels = json.load(f)

    preds = model.predict(X_test)
    y_pred = preds.argmax(axis=1)
    y_true = y_test.argmax(axis=1)

    print(classification_report(y_true, y_pred, target_names=labels))
    print('Confusion matrix:')
    print(confusion_matrix(y_true, y_pred))

if __name__ == '__main__':
    main()
