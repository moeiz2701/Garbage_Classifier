# knn_trainer.py
import time
import joblib
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from data_utils import load_data

def train_knn(X_train, y_train, X_val, y_val):
    model = KNeighborsClassifier(n_neighbors=3)
    start = time.time()
    model.fit(X_train, y_train)
    duration = time.time() - start
    joblib.dump(model, 'k-nn_garbage_classifier.pkl')
    y_pred = model.predict(X_val)
    acc = accuracy_score(y_val, y_pred) * 100
    print(f"k-NN Accuracy: {acc:.2f}% | Training Time: {duration:.2f}s")
    return acc, duration

def plot_learning_curve(X, y):
    from sklearn.model_selection import learning_curve
    train_sizes, train_scores, val_scores = learning_curve(
        KNeighborsClassifier(n_neighbors=3),
        X, y, cv=5, train_sizes=np.linspace(0.1, 1.0, 5), scoring='accuracy'
    )
    plt.plot(train_sizes, train_scores.mean(axis=1), label='Train')
    plt.plot(train_sizes, val_scores.mean(axis=1), label='Validation')
    plt.title("k-NN Learning Curve")
    plt.xlabel("Training Size")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid()
    plt.savefig("knn_learning_curve.png")
    plt.show()

def main():
    data_dir = "dataset/garbage_classification"
    X_train, y_train, X_val, y_val = load_data(data_dir)
    acc, time_taken = train_knn(X_train, y_train, X_val, y_val)
    plot_learning_curve(np.concatenate((X_train, X_val)), np.concatenate((y_train, y_val)))

if __name__ == "__main__":
    main()
