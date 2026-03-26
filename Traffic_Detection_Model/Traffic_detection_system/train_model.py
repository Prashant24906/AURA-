# train_model.py
"""Standalone training script with detailed metrics visualization."""

import matplotlib.pyplot as plt
import numpy as np
from traffic_classifier import TrafficClassifier
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


def plot_training_history(history):
    """Plot training and validation metrics."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    # Accuracy plot
    axes[0].plot(history.history['accuracy'], label='Train')
    axes[0].plot(history.history['val_accuracy'], label='Validation')
    axes[0].set_title('Model Accuracy')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Accuracy')
    axes[0].legend()
    axes[0].grid(True)
    
    # Loss plot
    axes[1].plot(history.history['loss'], label='Train')
    axes[1].plot(history.history['val_loss'], label='Validation')
    axes[1].set_title('Model Loss')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss')
    axes[1].legend()
    axes[1].grid(True)
    
    plt.tight_layout()
    plt.savefig('training_history.png')
    plt.show()


def plot_confusion_matrix(y_true, y_pred):
    """Plot confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(cm, display_labels=['Low Traffic', 'Heavy Traffic'])
    
    fig, ax = plt.subplots(figsize=(8, 6))
    disp.plot(ax=ax, cmap='Blues')
    plt.title('Confusion Matrix')
    plt.savefig('confusion_matrix.png')
    plt.show()


def main():
    # Initialize classifier
    classifier = TrafficClassifier(img_size=(128, 128))
    
    # Load data
    print("Loading dataset...")
    X, y = classifier.load_dataset('data')
    
    if len(X) == 0:
        print("No data found! Please organize your images as:")
        print("  data/heavy/  - images of heavy traffic")
        print("  data/low/    - images of low traffic")
        return
    
    print(f"Total images loaded: {len(X)}")
    print(f"Heavy traffic images: {np.sum(y == 1)}")
    print(f"Low traffic images: {np.sum(y == 0)}")
    
    # Build and train model
    classifier.build_model()
    print("\nModel Architecture:")
    classifier.model.summary()
    
    # Train with evaluation
    print("\nTraining model...")
    metrics, (X_train, X_test, y_train, y_test) = classifier.train(
        X, y, test_size=0.2, epochs=50, batch_size=32
    )
    
    # Plot results
    plot_training_history(classifier.history)
    plot_confusion_matrix(y_test, metrics['y_pred'])
    
    # Save model
    classifier.save_model('traffic_model.h5')
    
    print("\n" + "="*50)
    print("TRAINING COMPLETE")
    print("="*50)
    print(f"F1 Score: {metrics['f1_score']:.4f}")
    print(f"ROC-AUC: {metrics['roc_auc']:.4f}")
    print("Model saved as 'traffic_model.h5'")


if __name__ == "__main__":
    main()
