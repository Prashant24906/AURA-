# traffic_classifier.py
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, roc_auc_score, classification_report
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping
import os
import pickle
from collections import Counter


class TrafficClassifier:
    def __init__(self, img_size=(128, 128)):
        self.img_size = img_size
        self.model = None
        self.history = None
        self.class_names = ['low', 'heavy']  # 0: low traffic, 1: heavy traffic
        
    def preprocess_image(self, image_path):
        """Load and preprocess a single image."""
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        # Resize
        img = cv2.resize(img, self.img_size)
        
        # Convert to RGB (OpenCV loads as BGR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Apply image processing techniques
        img = self._enhance_image(img)
        
        # Normalize pixel values
        img = img / 255.0
        
        return img
    
    def _enhance_image(self, img):
        """Apply image processing for better feature extraction."""
        # Convert to LAB color space for better contrast
        lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        
        # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        
        # Merge channels back
        enhanced = cv2.merge([l, a, b])
        enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2RGB)
        
        return enhanced
    
    def load_dataset(self, data_dir):
        """Load images from directory structure: data_dir/heavy/, data_dir/low/"""
        images = []
        labels = []
        
        categories = {'heavy': 1, 'low': 0}
        
        for category, label in categories.items():
            category_path = os.path.join(data_dir, category)
            if not os.path.exists(category_path):
                print(f"Warning: {category_path} does not exist")
                continue
                
            image_files = [f for f in os.listdir(category_path) 
                          if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            
            print(f"Loading {len(image_files)} images from {category} class")
            
            for filename in image_files:
                img_path = os.path.join(category_path, filename)
                try:
                    img = self.preprocess_image(img_path)
                    images.append(img)
                    labels.append(label)
                except Exception as e:
                    print(f"Error loading {img_path}: {e}")
        
        if len(images) == 0:
            raise ValueError(f"No images found in {data_dir}")
        
        print(f"Total images loaded: {len(images)}")
        
        class_counts = Counter(labels)
        print(f"Class distribution: Heavy: {class_counts[1]}, Low: {class_counts[0]}")
        
        return np.array(images), np.array(labels)
    
    def build_model(self):
        """Build CNN model for traffic classification."""
        self.model = Sequential([
            Conv2D(32, (3, 3), activation='relu', input_shape=(*self.img_size, 3)),
            MaxPooling2D((2, 2)),
            
            Conv2D(64, (3, 3), activation='relu'),
            MaxPooling2D((2, 2)),
            
            Conv2D(128, (3, 3), activation='relu'),
            MaxPooling2D((2, 2)),
            
            Conv2D(128, (3, 3), activation='relu'),
            MaxPooling2D((2, 2)),
            
            Flatten(),
            Dense(256, activation='relu'),
            Dropout(0.5),
            Dense(128, activation='relu'),
            Dropout(0.3),
            Dense(1, activation='sigmoid')
        ])
        
        self.model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        return self.model
    
    def train(self, X, y, test_size=0.2, random_state=42, epochs=50, batch_size=32):
        """Train the model with train-test split."""
        
        class_counts = Counter(y)
        min_samples = min(class_counts.values())
        
        print(f"\nDataset Analysis:")
        print(f"Total samples: {len(X)}")
        print(f"Samples per class: Heavy={class_counts[1]}, Low={class_counts[0]}")
        
        if min_samples < 2:
            raise ValueError(f"Need at least 2 samples per class. Found Heavy={class_counts[1]}, Low={class_counts[0]}")
        
        # Perform train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        print(f"\nTraining samples: {len(X_train)}")
        print(f"Testing samples: {len(X_test)}")
        
        # Build model
        if self.model is None:
            self.build_model()
        
        # Data augmentation
        datagen = ImageDataGenerator(
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            horizontal_flip=True,
            zoom_range=0.2
        )
        
        # Early stopping
        early_stop = EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True
        )
        
        # Train
        print(f"\n🚀 Starting training for {epochs} epochs...")
        print("-" * 50)
        
        self.history = self.model.fit(
            datagen.flow(X_train, y_train, batch_size=batch_size),
            epochs=epochs,
            validation_data=(X_test, y_test),
            callbacks=[early_stop],
            verbose=1
        )
        
        # Evaluate
        print("\n" + "="*50)
        print("📊 MODEL EVALUATION")
        print("="*50)
        metrics = self.evaluate(X_test, y_test)
        
        return metrics, (X_train, X_test, y_train, y_test)
    
    def evaluate(self, X_test, y_test):
        """Evaluate model with F1 score and ROC-AUC."""
        y_pred_proba = self.model.predict(X_test)
        y_pred = (y_pred_proba > 0.5).astype(int).flatten()
        
        f1 = f1_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_pred_proba)
        
        print(f"\nF1 Score: {f1:.4f}")
        print(f"ROC-AUC Score: {roc_auc:.4f}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, target_names=self.class_names))
        
        return {
            'f1_score': f1,
            'roc_auc': roc_auc,
            'y_pred': y_pred,
            'y_pred_proba': y_pred_proba
        }
    
    def predict(self, image_path):
        """Predict traffic level for a single image."""
        if self.model is None:
            raise ValueError("Model not trained. Train or load a model first.")
        
        try:
            img = self.preprocess_image(image_path)
            img = np.expand_dims(img, axis=0)
            
            prediction = self.model.predict(img, verbose=0)[0][0]
            
            traffic_level = "HEAVY TRAFFIC" if prediction > 0.5 else "LOW TRAFFIC"
            confidence = prediction if prediction > 0.5 else 1 - prediction
            
            # Return 3 values as expected by main.py
            return traffic_level, confidence, prediction
            
        except Exception as e:
            raise Exception(f"Prediction failed: {str(e)}")
    
    def save_model(self, model_path='traffic_model.h5', class_names_path='class_names.pkl'):
        """Save the trained model and class names."""
        if self.model:
            self.model.save(model_path)
            print(f"✓ Model saved to {model_path}")
            
            with open(class_names_path, 'wb') as f:
                pickle.dump(self.class_names, f)
            print(f"✓ Class names saved to {class_names_path}")
    
    def load_model(self, model_path='traffic_model.h5', class_names_path='class_names.pkl'):
        """Load a pre-trained model and class names."""
        self.model = load_model(model_path)
        print(f"✓ Model loaded from {model_path}")
        
        if os.path.exists(class_names_path):
            with open(class_names_path, 'rb') as f:
                self.class_names = pickle.load(f)
            print(f"✓ Class names loaded from {class_names_path}")