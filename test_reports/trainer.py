import os
import json
import numpy as np
import random
import matplotlib.pyplot as plt
import seaborn as sns

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
from sklearn.metrics import (
    confusion_matrix, 
    classification_report, 
    precision_recall_fscore_support,
    roc_curve, auc, 
    precision_recall_curve, 
    average_precision_score
)

class HagridDataProcessor:
    def __init__(self, base_path):
        self.base_path = base_path
        self.gestures = ['palm', 'ok', 'fist', 'peace', 'peace_inverted', 'three2']
        self.landmarks = []
        self.labels = []
    
    def process_dataset(self, max_landmarks=21):
        for gesture in self.gestures:
            json_path = os.path.join(self.base_path, f'{gesture}.json')
            
            if not os.path.exists(json_path):
                print(f"No data found for {gesture}")
                continue
            
            with open(json_path, 'r') as f:
                data = json.load(f)
            
            for filename, entry in data.items():
                landmarks = entry.get('landmarks', [[]])[0]
                
                if len(landmarks) > max_landmarks:
                    landmarks = landmarks[:max_landmarks]
                elif len(landmarks) < max_landmarks:
                    landmarks += [[0, 0]] * (max_landmarks - len(landmarks))
                
                flat_landmarks = [coord for point in landmarks for coord in point]
                
                self.landmarks.append(flat_landmarks)
                self.labels.append(gesture)
        
        self.landmarks = np.array(self.landmarks)
        self.labels = np.array(self.labels)
        
        print(f"Total landmarks processed: {len(self.landmarks)}")
        print(f"Gesture distribution: {np.unique(self.labels, return_counts=True)}")
    
    def prepare_dataset(self, test_size=0.2, random_state=42):
        if len(self.landmarks) == 0:
            raise ValueError("No landmarks found. Make sure to run process_dataset() first.")
        
        label_encoder = LabelEncoder()
        encoded_labels = label_encoder.fit_transform(self.labels)
        
        encoded_labels = to_categorical(encoded_labels)
        
        X_train, X_test, y_train, y_test = train_test_split(
            self.landmarks, 
            encoded_labels, 
            test_size=test_size, 
            random_state=random_state,
            stratify=encoded_labels
        )
        
        return X_train, X_test, y_train, y_test, label_encoder
    
    def create_cnn_model(self, input_shape, num_classes):
        model = Sequential([
            Conv1D(64, kernel_size=3, activation='relu', input_shape=input_shape),
            BatchNormalization(),
            MaxPooling1D(pool_size=2),
            
            Conv1D(128, kernel_size=3, activation='relu'),
            BatchNormalization(),
            MaxPooling1D(pool_size=2),
            
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
        X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
        X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)
        
        model = self.create_cnn_model(
            input_shape=(X_train.shape[1], 1), 
            num_classes=y_train.shape[1]
        )
        
        early_stopping = EarlyStopping(
            monitor='val_loss', 
            patience=10, 
            restore_best_weights=True
        )
        
        reduce_lr = ReduceLROnPlateau(
            monitor='val_loss', 
            factor=0.2, 
            patience=5, 
            min_lr=0.00001
        )
        
        history = model.fit(
            X_train, y_train,
            validation_data=(X_test, y_test),
            epochs=100,
            batch_size=32,
            callbacks=[early_stopping, reduce_lr],
            verbose=1
        )
        

        self.plot_training_history(history)
        
        return model, history
    
    def plot_training_history(self, history):
        plt.figure(figsize=(12, 5))
       
        plt.subplot(1, 2, 1)
        plt.plot(history.history['accuracy'], label='Train Accuracy')
        plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
        plt.title('Model Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        
        
        plt.subplot(1, 2, 2)
        plt.plot(history.history['loss'], label='Train Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.title('Model Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig('training_history.png')
        plt.close()
    
    def evaluate_model(self, model, X_test, y_test):
        X_test_reshaped = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)
        
        
        test_loss, test_accuracy = model.evaluate(X_test_reshaped, y_test)
        print(f"Test Loss: {test_loss:.4f}")
        print(f"Test Accuracy: {test_accuracy * 100:.2f}%")
        
    
        y_pred_prob = model.predict(X_test_reshaped)
        y_pred_classes = np.argmax(y_pred_prob, axis=1)
        y_true_classes = np.argmax(y_test, axis=1)
        
        print("\nDetailed Classification Report:")
        report = classification_report(
            y_true_classes, 
            y_pred_classes, 
            target_names=self.gestures,
            output_dict=True
        )
        print(classification_report(
            y_true_classes, 
            y_pred_classes, 
            target_names=self.gestures
        ))
        
        self.plot_confusion_matrix(y_true_classes, y_pred_classes)
        
        precision, recall, f1, support = precision_recall_fscore_support(
            y_true_classes, 
            y_pred_classes, 
            average='weighted'
        )
        
        print("\nOverall Performance Metrics:")
        print(f"Precision (weighted): {precision:.4f}")
        print(f"Recall (weighted): {recall:.4f}")
        print(f"F1 Score (weighted): {f1:.4f}")
        
        self.plot_class_metrics(report)
        s
        self.plot_roc_curves(y_test, y_pred_prob)
        
        
        self.plot_precision_recall_curves(y_test, y_pred_prob)
        
      
        self.save_metrics(report, precision, recall, f1, test_accuracy)
    
    def plot_confusion_matrix(self, y_true, y_pred):
        plt.figure(figsize=(10, 8))
        
       
        cm = confusion_matrix(y_true, y_pred)
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        sns.heatmap(
            cm_normalized, 
            annot=True, 
            fmt='.2f', 
            cmap='Blues',
            xticklabels=self.gestures,
            yticklabels=self.gestures
        )
        
        plt.title('Normalized Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig('confusion_matrix.png')
        plt.close()
    
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            cm, 
            annot=True, 
            fmt='d', 
            cmap='Blues',
            xticklabels=self.gestures,
            yticklabels=self.gestures
        )
        
        plt.title('Confusion Matrix (counts)')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig('confusion_matrix_counts.png')
        plt.close()
    
    def plot_class_metrics(self, report):
      
        classes = self.gestures
        precision = [report[cls]['precision'] for cls in classes]
        recall = [report[cls]['recall'] for cls in classes]
        f1 = [report[cls]['f1-score'] for cls in classes]
        
        plt.figure(figsize=(12, 6))
        x = np.arange(len(classes))
        width = 0.25
        
        plt.bar(x - width, precision, width, label='Precision')
        plt.bar(x, recall, width, label='Recall')
        plt.bar(x + width, f1, width, label='F1-score')
        
        plt.xlabel('Gesture Classes')
        plt.ylabel('Score')
        plt.title('Performance Metrics by Class')
        plt.xticks(x, classes, rotation=45)
        plt.legend()
        plt.tight_layout()
        plt.savefig('class_metrics.png')
        plt.close()
    
    def plot_roc_curves(self, y_test, y_pred_prob):
        n_classes = len(self.gestures)
        
        plt.figure(figsize=(12, 8))
        
        for i in range(n_classes):
         
            fpr, tpr, _ = roc_curve(y_test[:, i], y_pred_prob[:, i])
            roc_auc = auc(fpr, tpr)
            
            plt.plot(
                fpr, 
                tpr, 
                lw=2,
                label=f'ROC curve for {self.gestures[i]} (area = {roc_auc:.2f})'
            )
        
        plt.plot([0, 1], [0, 1], 'k--', lw=2)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curves (One-vs-Rest)')
        plt.legend(loc="lower right")
        plt.tight_layout()
        plt.savefig('roc_curves.png')
        plt.close()
    
    def plot_precision_recall_curves(self, y_test, y_pred_prob):
        n_classes = len(self.gestures)
        
        plt.figure(figsize=(12, 8))
        
        for i in range(n_classes):
            # Compute precision-recall curve
            precision, recall, _ = precision_recall_curve(y_test[:, i], y_pred_prob[:, i])
            avg_precision = average_precision_score(y_test[:, i], y_pred_prob[:, i])
            
            plt.plot(
                recall, 
                precision,
                lw=2, 
                label=f'Precision-Recall for {self.gestures[i]} (AP = {avg_precision:.2f})'
            )
        
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curves')
        plt.legend(loc="best")
        plt.tight_layout()
        plt.savefig('precision_recall_curves.png')
        plt.close()
    
    def save_metrics(self, report, precision, recall, f1, accuracy):
        
        metrics = {
            'overall': {
                'accuracy': float(accuracy),
                'precision': float(precision),
                'recall': float(recall),
                'f1_score': float(f1)
            },
            'per_class': report
        }
        
        with open('model_metrics.json', 'w') as f:
            json.dump(metrics, f, indent=4)
        
        print("\nAll metrics have been saved to 'model_metrics.json'")

def main():
    base_path = r'C:\Users\reshm\OneDrive\Pictures\MINI PROJECT\datasets'
    
    processor = HagridDataProcessor(base_path)
    processor.process_dataset()
    
    X_train, X_test, y_train, y_test, label_encoder = processor.prepare_dataset()
    
    model, history = processor.train_model(X_train, X_test, y_train, y_test)
    
    processor.evaluate_model(model, X_test, y_test)
    
    model.save('model.h5')
    print("Model saved as 'model.h5'")
    

    with open('model_summary.txt', 'w') as f:
        model.summary(print_fn=lambda x: f.write(x + '\n'))
    print("Model summary saved as 'model_summary.txt'")

if __name__ == '__main__':
    main()