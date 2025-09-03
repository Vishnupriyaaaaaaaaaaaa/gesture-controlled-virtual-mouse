import os
import json
import numpy as np
import random

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv1D, MaxPooling1D, Flatten, 
    Dense, Dropout, BatchNormalization
)
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

class HagridDataProcessor:
    def __init__(self, base_path):
        """
        Initialize Hagrid dataset processor
        
        Args:
            base_path (str): Base directory containing gesture JSON files
        """
        self.base_path = base_path
        # Updated gestures to match your dataset
        self.gestures = ['palm', 'ok', 'fist', 'peace', 'peace_inverted', 'three2']
        self.landmarks = []
        self.labels = []
    
    def process_dataset(self, max_landmarks=21):
        """
        Process JSON files for each gesture and extract landmarks
        
        Args:
            max_landmarks (int): Maximum number of landmarks to process
        """
        for gesture in self.gestures:
            json_path = os.path.join(self.base_path, f'{gesture}.json')
            
            if not os.path.exists(json_path):
                print(f"Warning: No data found for {gesture} at {json_path}")
                continue
            
            try:
                with open(json_path, 'r') as f:
                    data = json.load(f)
                
                print(f"Processing {gesture}: {len(data)} samples")
                
                # Process each entry in the single gesture JSON
                for filename, entry in data.items():
                    # Extract landmarks
                    landmarks = entry.get('landmarks', [[]])[0]
                    
                    # Skip empty landmarks
                    if not landmarks or len(landmarks) == 0:
                        continue
                    
                    # Standardize landmark count
                    if len(landmarks) > max_landmarks:
                        landmarks = landmarks[:max_landmarks]
                    elif len(landmarks) < max_landmarks:
                        landmarks += [[0, 0]] * (max_landmarks - len(landmarks))
                    
                    # Flatten landmarks
                    flat_landmarks = [coord for point in landmarks for coord in point]
                    
                    self.landmarks.append(flat_landmarks)
                    self.labels.append(gesture)
                    
            except Exception as e:
                print(f"Error processing {gesture}.json: {e}")
                continue
        
        # Convert to numpy arrays
        self.landmarks = np.array(self.landmarks)
        self.labels = np.array(self.labels)
        
        # Print dataset info
        print(f"\nDataset Summary:")
        print(f"Total landmarks processed: {len(self.landmarks)}")
        if len(self.labels) > 0:
            unique_labels, counts = np.unique(self.labels, return_counts=True)
            for label, count in zip(unique_labels, counts):
                print(f"  {label}: {count} samples")
        else:
            print("Warning: No data processed!")
    
    def prepare_dataset(self, test_size=0.2, random_state=42):
        """
        Prepare dataset for training
        
        Args:
            test_size (float): Proportion of data to use for testing
            random_state (int): Random seed for reproducibility
        
        Returns:
            Processed training and testing datasets
        """
        # Ensure we have data
        if len(self.landmarks) == 0:
            raise ValueError("No landmarks found. Make sure to run process_dataset() first.")
        
        # Encode labels
        label_encoder = LabelEncoder()
        encoded_labels = label_encoder.fit_transform(self.labels)
        
        # One-hot encode labels
        encoded_labels = to_categorical(encoded_labels)
        
        print(f"Number of classes: {len(label_encoder.classes_)}")
        print(f"Class names: {label_encoder.classes_}")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            self.landmarks, 
            encoded_labels, 
            test_size=test_size, 
            random_state=random_state,
            stratify=encoded_labels
        )
        
        print(f"Training samples: {len(X_train)}")
        print(f"Testing samples: {len(X_test)}")
        
        return X_train, X_test, y_train, y_test, label_encoder
    
    def create_cnn_model(self, input_shape, num_classes):
        """
        Create CNN model for gesture recognition
        
        Args:
            input_shape (tuple): Shape of input data
            num_classes (int): Number of gesture classes
        
        Returns:
            Compiled Keras model
        """
        model = Sequential([
            # 1D Convolutional layers
            Conv1D(64, kernel_size=3, activation='relu', input_shape=input_shape),
            BatchNormalization(),
            MaxPooling1D(pool_size=2),
            
            Conv1D(128, kernel_size=3, activation='relu'),
            BatchNormalization(),
            MaxPooling1D(pool_size=2),
            
            # Flatten and dense layers
            Flatten(),
            Dense(128, activation='relu'),
            Dropout(0.5),
            Dense(64, activation='relu'),
            Dropout(0.3),
            Dense(num_classes, activation='softmax')
        ])
        
        model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def train_model(self, X_train, X_test, y_train, y_test):
        """
        Train the gesture recognition model
        
        Args:
            X_train, X_test: Training and testing landmark data
            y_train, y_test: Training and testing labels
        
        Returns:
            Trained model and training history
        """
        # Reshape input for 1D CNN
        X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
        X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)
        
        print(f"Input shape for training: {X_train.shape}")
        
        # Create model
        model = self.create_cnn_model(
            input_shape=(X_train.shape[1], 1), 
            num_classes=y_train.shape[1]
        )
        
        # Print model summary
        print("\nModel Architecture:")
        model.summary()
        
        # Callbacks
        early_stopping = EarlyStopping(
            monitor='val_loss', 
            patience=10, 
            restore_best_weights=True,
            verbose=1
        )
        
        reduce_lr = ReduceLROnPlateau(
            monitor='val_loss', 
            factor=0.2, 
            patience=5, 
            min_lr=0.00001,
            verbose=1
        )
        
        # Train model
        print("\nStarting training...")
        history = model.fit(
            X_train, y_train,
            validation_data=(X_test, y_test),
            epochs=100,
            batch_size=32,
            callbacks=[early_stopping, reduce_lr],
            verbose=1
        )
        
        return model, history
    
    def evaluate_model(self, model, X_test, y_test):
        """
        Evaluate model performance
        
        Args:
            model: Trained Keras model
            X_test: Test landmark data
            y_test: Test labels
        """
        # Reshape test data
        X_test_reshaped = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)
        
        # Evaluate
        test_loss, test_accuracy = model.evaluate(X_test_reshaped, y_test, verbose=0)
        print(f"\nFinal Model Performance:")
        print(f"Test Loss: {test_loss:.4f}")
        print(f"Test Accuracy: {test_accuracy * 100:.2f}%")
        
        # Detailed classification report
        y_pred = model.predict(X_test_reshaped, verbose=0)
        y_pred_classes = np.argmax(y_pred, axis=1)
        y_true_classes = np.argmax(y_test, axis=1)
        
        # Classification report
        try:
            from sklearn.metrics import classification_report
            print("\nDetailed Classification Report:")
            print(classification_report(
                y_true_classes, 
                y_pred_classes, 
                target_names=self.gestures
            ))
        except ImportError:
            print("sklearn not available for detailed metrics")

def main():
    # Use the correct path to your dataset
    # Go back one directory from src to find datasets
    base_path = os.path.join('..', 'datasets')
    
    print("Gesture Recognition Model Training")
    print("=" * 40)
    
    # Check if dataset directory exists
    if not os.path.exists(base_path):
        print(f"Error: Dataset directory '{base_path}' not found!")
        print("Please create the datasets folder and add gesture JSON files.")
        return
    
    try:
        # Process dataset
        processor = HagridDataProcessor(base_path)
        processor.process_dataset()
        
        # Check if any data was loaded
        if len(processor.landmarks) == 0:
            print("Error: No gesture data found!")
            print("Please ensure your JSON files contain valid landmark data.")
            return
        
        # Prepare data
        X_train, X_test, y_train, y_test, label_encoder = processor.prepare_dataset()
        
        # Train model
        model, history = processor.train_model(X_train, X_test, y_train, y_test)
        
        # Evaluate model
        processor.evaluate_model(model, X_test, y_test)
        
        # Save model
        model_path = os.path.join('..', 'models', 'gesture_model.h5')
        os.makedirs(os.path.join('..', 'models'), exist_ok=True)
        model.save(model_path)
        print(f"\nModel saved to: {model_path}")
        
        # Save label encoder for later use
        import pickle
        with open(os.path.join('..', 'models', 'label_encoder.pkl'), 'wb') as f:
            pickle.dump(label_encoder, f)
        
        print("Training completed successfully!")
        
    except Exception as e:
        print(f"Error during training: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main()