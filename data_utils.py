# data_utils.py
import os
import numpy as np
from PIL import Image
from glob import glob
from skimage.feature import hog

IMG_SIZE = 128
CLASS_NAMES = ['battery', 'biological', 'brown-glass', 'cardboard', 'clothes',
               'green-glass', 'metal', 'paper', 'plastic', 'shoes', 'trash', 'white-glass']

def compute_color_histogram(img, bins=32):
    if img.mode != 'RGB':
        img = img.convert('RGB')
    img_array = np.array(img)
    hist_r = np.histogram(img_array[:, :, 0], bins=bins, range=(0, 256))[0]
    hist_g = np.histogram(img_array[:, :, 1], bins=bins, range=(0, 256))[0]
    hist_b = np.histogram(img_array[:, :, 2], bins=bins, range=(0, 256))[0]
    hist = np.concatenate([hist_r, hist_g, hist_b])
    return hist / (hist.sum() + 1e-6)

def load_data(data_dir, subset_ratio=0.8, seed=42):
    X, y = [], []
    for class_idx, class_name in enumerate(CLASS_NAMES):
        paths = glob(os.path.join(data_dir, class_name, '*.jpg'))
        for path in paths:
            img_rgb = Image.open(path).resize((IMG_SIZE, IMG_SIZE))
            img_gray = img_rgb.convert('L')
            img_array = np.array(img_gray) / 255.0
            hog_features = hog(img_array, pixels_per_cell=(16, 16), cells_per_block=(2, 2),
                               orientations=9, feature_vector=True)
            color_features = compute_color_histogram(img_rgb)
            features = np.concatenate([hog_features, color_features])
            X.append(features)
            y.append(class_idx)
    X, y = np.array(X), np.array(y)
    np.random.seed(seed)
    indices = np.random.permutation(len(X))
    split = int(len(X) * subset_ratio)
    return X[indices[:split]], y[indices[:split]], X[indices[split:]], y[indices[split:]]
