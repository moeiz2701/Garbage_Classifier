# svm_trainer.py
import time
import joblib
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from data_utils import load_data

def train_svm(X_train, y_train, X_val, y_val):
    model = SVC(kernel='rbf', C=10, class_weight='balanced')
    start = time.time()
    model.fit(X_train, y_train)
    duration = time.time() - start
    joblib.dump(model, 'svm_garbage_classifier.pkl')
    y_pred = model.predict(X_val)
    acc = accuracy_score(y_val, y_pred) * 100
    print(f"SVM Accuracy: {acc:.2f}% | Training Time: {duration:.2f}s")
    return acc, duration

def plot_learning_curve(X, y):
    from sklearn.model_selection import learning_curve
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    train_sizes, train_scores, val_scores = learning_curve(
        SVC(kernel='rbf', C=10, class_weight='balanced'),
        X, y, cv=5, train_sizes=np.linspace(0.1, 1.0, 5), scoring='accuracy'
    )
    train_mean = train_scores.mean(axis=1)
    val_mean = val_scores.mean(axis=1)

    plt.plot(train_sizes, train_mean, label='Train')
    plt.plot(train_sizes, val_mean, label='Validation')
    plt.xlabel('Training Size')
    plt.ylabel('Accuracy')
    plt.title('SVM Learning Curve')
    plt.legend()
    plt.grid()
    plt.savefig("svm_learning_curve.png")
    plt.close()

def main():
    data_dir = "dataset/garbage_classification"
    X_train, y_train, X_val, y_val = load_data(data_dir)
    acc, time_taken = train_svm(X_train, y_train, X_val, y_val)
    plot_learning_curve(np.concatenate((X_train, X_val)), np.concatenate((y_train, y_val)))

if __name__ == "__main__":
    main()
