# rf_trainer.py
import time
import joblib
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from data_utils import load_data


def train_rf(X_train, y_train, X_val, y_val):
    model = RandomForestClassifier(n_estimators=200, class_weight='balanced',
                                   random_state=42)
    start = time.time()
    model.fit(X_train, y_train)
    duration = time.time() - start
    joblib.dump(model, 'random_forest_garbage_classifier.pkl')
    y_pred = model.predict(X_val)
    acc = accuracy_score(y_val, y_pred) * 100
    print(f"Random Forest Accuracy: {acc:.2f}% | "
          f"Training Time: {duration:.2f}s")
    return acc, duration


def plot_learning_curve(X, y):
    from sklearn.model_selection import learning_curve
    train_sizes, train_scores, val_scores = learning_curve(
        RandomForestClassifier(n_estimators=200, class_weight='balanced'),
        X, y, cv=5, train_sizes=np.linspace(0.1, 1.0, 5),
        scoring='accuracy'
    )
    plt.plot(train_sizes, train_scores.mean(axis=1), label='Train')
    plt.plot(train_sizes, val_scores.mean(axis=1), label='Validation')
    plt.title("Random Forest Learning Curve")
    plt.xlabel("Training Size")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid()
    plt.savefig("rf_learning_curve.png")
    plt.show()


def main():
    data_dir = "dataset/garbage_classification"
    X_train, y_train, X_val, y_val = load_data(data_dir)
    acc, time_taken = train_rf(X_train, y_train, X_val, y_val)
    plot_learning_curve(np.concatenate((X_train, X_val)),
                        np.concatenate((y_train, y_val)))


if __name__ == "__main__":
    main()
