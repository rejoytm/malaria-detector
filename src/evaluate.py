import cv2
import numpy as np
import tensorflow as tf

from src import config
from src.data_loader import load_dataset_splits, load_dataset_splits_dual

# Calculates precision, recall, F1 score, and confusion matrix values
def calculate_metrics(y_true, y_pred):
    cm = tf.math.confusion_matrix(y_true, y_pred)

    # Instead of tn, fp, fn, tp = cm[0][0], cm[0][1], cm[1][0], cm[1][1]
    # we swap the roles of positive and negative classes to make 
    # "Parasitic" the positive class and "Uninfected" the negative class
    tp, fn, fp, tn = cm[0][0], cm[0][1], cm[1][0], cm[1][1]

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1_score = 2 * (precision * recall) / (precision + recall + 1e-7)

    metrics_dict = {
        'tp': tp,
        'tn': tn,
        'fp': fp,
        'fn': fn,
        'precision': precision,
        'recall': recall,
        'f1_score': f1_score
    }

    return metrics_dict

# Preprocess a single image (resize, normalize, and add batch dimension)
def preprocess_image(img_path):
    img = cv2.imread(img_path)
    if img is None:
        raise FileNotFoundError(f"Image not found at path: {img_path}")
    
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, config.IMG_SIZE)
    img_array = np.asarray(img, dtype=np.float32) / 255.0
    
    # Add batch dimension
    input_tensor = np.expand_dims(img_array, axis=0)
    return input_tensor

def load_model(model_path):
    try:
        model = tf.keras.models.load_model(model_path)
        print(f"Model loaded successfully from {model_path}.")
        return model
    except Exception as e:
        print(f"Failed to load model from {model_path}: {e}")
        return None

# Loads the model, makes predictions, and calculates metrics
def evaluate_model(model_path, data_path):
    model = load_model(model_path)
    if model is None:
        raise ValueError(f"Model could not be loaded from {model_path}.")
    
    train_ds, val_ds, test_ds = load_dataset_splits(data_path)
    
    y_true = []
    y_pred = []
    
    for images, labels in test_ds:
        predictions = model.predict(images)
        predicted_labels = (predictions > 0.5).astype(int)
        
        y_true.extend(labels.numpy().flatten())
        y_pred.extend(predicted_labels.flatten())

    # Calculate metrics
    metrics = calculate_metrics(np.array(y_true), np.array(y_pred))
    
    # Print and return the evaluation metrics
    print(f"Evaluation Metrics for model at {model_path}:")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"F1 Score: {metrics['f1_score']:.4f}")
    print(f"True Positives (TP): {metrics['tp']}")
    print(f"True Negatives (TN): {metrics['tn']}")
    print(f"False Positives (FP): {metrics['fp']}")
    print(f"False Negatives (FN): {metrics['fn']}")
    
    return metrics

# Loads the fusion model, makes predictions, and calculates metrics
def evaluate_fusion_model(model_path, original_data_path, normalized_data_path):
    model = load_model(model_path)
    if model is None:
        raise ValueError(f"Fusion model could not be loaded from {model_path}.")
    
    # Load dual dataset
    train_ds, val_ds, test_ds = load_dataset_splits_dual(original_data_path, normalized_data_path)
    
    y_true = []
    y_pred = []
    
    # Inputs will be a tuple (original_images, normalized_images)
    for inputs, labels in test_ds:
        # Use the dual inputs for prediction
        predictions = model.predict(inputs)
        predicted_labels = (predictions > 0.5).astype(int)
        
        y_true.extend(labels.numpy().flatten())
        y_pred.extend(predicted_labels.flatten())

    # Calculate metrics
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    metrics = calculate_metrics(y_true, y_pred)
    
    # Print and return the evaluation metrics
    print(f"Evaluation Metrics for model at {model_path}:")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"F1 Score: {metrics['f1_score']:.4f}")
    print(f"True Positives (TP): {metrics['tp']}")
    print(f"True Negatives (TN): {metrics['tn']}")
    print(f"False Positives (FP): {metrics['fp']}")
    print(f"False Negatives (FN): {metrics['fn']}")
    
    return metrics

if __name__ == "__main__":
    for model in config.EVALUATION_MODEL_PATHS:
        evaluate_model(config.EVALUATION_MODEL_PATHS[model], config.EVALUATION_DATA_DIR)