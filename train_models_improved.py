import os
import time
import numpy as np
from PIL import Image
from skimage.feature import hog
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from glob import glob
import joblib

# Constants
IMG_SIZE = 128
NUM_CLASSES = 12
CLASS_NAMES = ['battery', 'biological', 'brown-glass', 'cardboard', 'clothes',
               'green-glass', 'metal', 'paper', 'plastic', 'shoes', 'trash',
               'white-glass']


def check_class_distribution(data_dir, class_names):
    class_counts = {}
    for class_name in class_names:
        class_path = os.path.join(data_dir, class_name, '*.jpg')
        img_paths = glob(class_path)
        class_counts[class_name] = len(img_paths)
    print("\nClass Distribution:")
    for class_name, count in class_counts.items():
        print(f"{class_name:<15} {count} images")
    return class_counts


def compute_color_histogram(img, bins=32):
    if img.mode != 'RGB':
        img = img.convert('RGB')
    img_array = np.array(img)
    hist_r = np.histogram(img_array[:, :, 0], bins=bins, range=(0, 256))[0]
    hist_g = np.histogram(img_array[:, :, 1], bins=bins, range=(0, 256))[0]
    hist_b = np.histogram(img_array[:, :, 2], bins=bins, range=(0, 256))[0]
    hist = np.concatenate([hist_r, hist_g, hist_b])
    return hist / (hist.sum() + 1e-6)


def load_and_extract_features(data_dir, subset_ratio=0.8):
    X, y = [], []
    for class_idx, class_name in enumerate(CLASS_NAMES):
        class_path = os.path.join(data_dir, class_name, '*.jpg')
        img_paths = glob(class_path)
        if not img_paths:
            print(f"Warning: No images found in {class_path}")
        for img_path in img_paths:
            img_rgb = Image.open(img_path).resize((IMG_SIZE, IMG_SIZE))
            img_gray = img_rgb.convert('L')
            img_array = np.array(img_gray) / 255.0
            hog_features = hog(img_array, pixels_per_cell=(16, 16),
                               cells_per_block=(2, 2), orientations=9,
                               feature_vector=True)
            color_features = compute_color_histogram(img_rgb)
            features = np.concatenate([hog_features, color_features])
            X.append(features)
            y.append(class_idx)
    if not X:
        raise ValueError("No images loaded. Check dataset directory structure.")
    X, y = np.array(X), np.array(y)
    np.random.seed(42)
    indices = np.random.permutation(len(X))
    train_size = int(len(X) * subset_ratio)
    train_idx, val_idx = indices[:train_size], indices[train_size:]
    return X[train_idx], y[train_idx], X[val_idx], y[val_idx]


def tune_and_train_models(X_train, y_train, X_val, y_val):
    # Scale features for SVM and k-NN
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    # SVM Hyperparameter Tuning
    svm_param_grid = {
        'C': [0.1, 1, 10, 100],
        'gamma': [0.001, 0.01, 0.1, 1],
        'kernel': ['rbf', 'linear']
    }
    svm_grid_search = GridSearchCV(SVC(class_weight='balanced',
                                       random_state=42), svm_param_grid,
                                   cv=3, verbose=1, n_jobs=-1)
    svm_grid_search.fit(X_train_scaled, y_train)
    print(f"SVM Best Parameters: {svm_grid_search.best_params_}")
    best_svm = svm_grid_search.best_estimator_

    # Random Forest Hyperparameter Tuning
    rf_param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10]
    }
    rf_grid_search = GridSearchCV(RandomForestClassifier(
                                      class_weight='balanced',
                                      random_state=42, n_jobs=-1),
                                  rf_param_grid, cv=3, verbose=1,
                                  n_jobs=-1)
    rf_grid_search.fit(X_train_scaled, y_train)
    print(f"Random Forest Best Parameters: {rf_grid_search.best_params_}")
    best_rf = rf_grid_search.best_estimator_

    # k-NN Hyperparameter Tuning
    knn_param_grid = {
        'n_neighbors': [3, 5, 7],
        'metric': ['euclidean', 'manhattan']
    }
    knn_grid_search = GridSearchCV(KNeighborsClassifier(), knn_param_grid,
                                   cv=3, verbose=1, n_jobs=-1)
    knn_grid_search.fit(X_train_scaled, y_train)
    print(f"k-NN Best Parameters: {knn_grid_search.best_params_}")
    best_knn = knn_grid_search.best_estimator_

    # Evaluate models
    models = {'SVM': best_svm, 'Random Forest': best_rf, 'k-NN': best_knn}
    results = {'Model': [], 'Accuracy (%)': [], 'Training Time (s)': []}

    for name, model in models.items():
        print(f"\nTraining {name}...")
        start_time = time.time()
        model.fit(X_train_scaled, y_train)
        training_time = time.time() - start_time
        # Save model to file
        filename = (f"{name.replace(' ', '_').lower()}_improved_garbage_"
                    f"classifier_tuned.pkl")
        joblib.dump(model, filename)
        print(f"{name} model saved to: {filename}")
        # Evaluate model
        y_pred = model.predict(X_val_scaled)
        accuracy = accuracy_score(y_val, y_pred) * 100
        results['Model'].append(name)
        results['Accuracy (%)'].append(accuracy)
        results['Training Time (s)'].append(training_time)
        print(f"{name} - Validation Accuracy: {accuracy:.2f}%, "
              f"Training Time: {training_time:.2f}s")

    return results

def visualize_results(results):
    plt.figure(figsize=(8, 5))
    plt.bar(results['Model'], results['Accuracy (%)'],
             color=['blue', 'green', 'orange'])
    plt.title('Model Validation Accuracies')
    plt.ylabel('Accuracy (%)')
    plt.ylim(0, 100)
    for i, acc in enumerate(results['Accuracy (%)']):
        plt.text(i, acc + 1, f'{acc:.2f}%', ha='center')
    plt.savefig('model_accuracies_tuned.png')
    plt.show()

    print("\nModel Comparison Table:")
    for i in range(len(results['Model'])):
        print(f"{results['Model'][i]:<15} "
              f"{results['Accuracy (%)'][i]:<15.2f} "
              f"{results['Training Time (s)'][i]:<15.2f}")


def main():
    data_dir = os.path.join("dataset", "garbage_classification")
    try:
        check_class_distribution(data_dir, CLASS_NAMES)
        X_train, y_train, X_val, y_val = load_and_extract_features(data_dir)
        print(f"\nLoaded {len(X_train)} training samples and "
              f"{len(X_val)} validation samples.")
    except Exception as e:
        print(f"Error loading data: {e}")
        raise

    results = tune_and_train_models(X_train, y_train, X_val, y_val)
    visualize_results(results)


if __name__ == "__main__":
    main()
